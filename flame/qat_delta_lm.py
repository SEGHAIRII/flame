# -*- coding: utf-8 -*-
"""
QAT finetuning for SparseMamba2DeltaForCausalLM with channel-wise thresholds.

Phase
-----
- Load a pretrained SparseMamba2DeltaForCausalLM checkpoint.
  Supports both TorchTitan DCP directories and plain .pt files.
- Calibrate per-channel thresholds from a small fixed subset of the data.
- Fine-tune with quantization ON (apply_qat_quantization=True):
    loss = cross_entropy + l1_weight * L1(discrete_temporal_diffs)
- Save final checkpoint + thresholds.

Checkpoint formats supported
-----------------------------
  DCP directory   : pass the directory path  (TorchTitan / torchrun default)
  Plain .pt file  : {"model_state_dict": ...} | {"model_state_dicts": [...]} | raw state_dict

Usage (single node, 4 GPUs)
----------------------------
torchrun --nproc_per_node 4 qat_finetune_lm.py \
    --checkpoint path/to/dcp_dir_or_file.pt \
    --config_path path/to/config.json \
    --output_dir ./qat_lm_ckpts \
    --dataset HuggingFaceFW/fineweb-edu \
    --dataset_name sample-100BT \
    --dataset_split train \
    --tokenizer_path fla-hub/transformer-1.3B-100B \
    --steps 5000 --batch_size 4 --lr 1e-5 --threshold_lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import sparse_mamba.custom_models  # noqa: F401
from flame.data import build_dataloader_causal, build_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# DDP helpers
# ============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return dist.get_rank() == 0


# ============================================================================
# L1-weight scheduler (cosine ramp-up)
# ============================================================================

def schedule_value(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine ramp from `start` to `end` over `total_steps`."""
    if total_steps <= 0 or step >= total_steps:
        return end
    return start + (end - start) * 0.5 * (1.0 - math.cos(step / total_steps * math.pi))


# ============================================================================
# Datasets
# ============================================================================

def build_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _calibration_batch_limit(args) -> int:
    batch_cap  = args.calibration_batches
    sample_cap = (
        math.ceil(args.calibration_samples / args.batch_size)
        if args.calibration_samples > 0
        else 0
    )
    if batch_cap <= 0 and sample_cap <= 0:
        return 0
    if batch_cap <= 0:
        return sample_cap
    if sample_cap <= 0:
        return batch_cap
    return min(batch_cap, sample_cap)


def build_dataloaders(args, world_size: int, global_rank: int):
    tokenizer = build_tokenizer(args.tokenizer_path)
    dataset_kwargs = {
        "dataset":       args.dataset,
        "dataset_name":  args.dataset_name,
        "dataset_split": args.dataset_split,
        "streaming":     True,
        "num_workers":   args.num_workers,
        "seed":          args.seed,
    }

    train_ds = build_dataset(dp_degree=world_size, **dataset_kwargs)
    train_loader = build_dataloader_causal(
        dataset=train_ds,
        tokenizer=tokenizer,
        rank=global_rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        world_size=world_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Rank 0 uses a separate loader for calibration so the training stream stays
    # untouched and aligned across ranks after threshold broadcast.
    calib_loader = None
    if global_rank == 0:
        calib_ds = build_dataset(dp_degree=1, **dataset_kwargs)
        calib_loader = build_dataloader_causal(
            dataset=calib_ds,
            tokenizer=tokenizer,
            rank=0,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            world_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    return train_loader, calib_loader, tokenizer


# ============================================================================
# Model loading  —  DCP directory  OR  plain .pt file
# ============================================================================

def load_config(config_path: str, initial_threshold: Optional[float] = None):
    config = AutoConfig.from_pretrained(config_path)
    if initial_threshold is not None:
        config.threshold = float(initial_threshold)
    # QAT phase: no internal regularization term; we compute L1 externally
    config.regularization_mode = "none"
    return config


def _drop_threshold_keys(state_dict: dict) -> list:
    """Remove threshold raw keys so calibration can reinitialise them cleanly."""
    dropped = [k for k in list(state_dict.keys()) if k.endswith("_threshold_raw")]
    for k in dropped:
        state_dict.pop(k)
    return dropped


def _load_from_dcp(model, checkpoint_path: str):
    """
    Load a TorchTitan DCP checkpoint directory.
    TorchTitan wraps the model state dict under the "model" key.
    """
    if is_main():
        print(f"  Format : TorchTitan DCP directory")

    model_sd = model.state_dict()
    storage  = {"model": model_sd}
    dcp.load(storage, checkpoint_id=checkpoint_path)

    loaded_sd = storage["model"]
    dropped   = _drop_threshold_keys(loaded_sd)
    missing, unexpected = model.load_state_dict(loaded_sd, strict=False)

    if is_main():
        if dropped:    print(f"  Dropped {len(dropped)} threshold keys for fresh calibration.")
        if missing:    print(f"  Missing keys    : {len(missing)}")
        if unexpected: print(f"  Unexpected keys : {len(unexpected)}")

    return model


def _load_from_pt(model, checkpoint_path: str):
    """
    Load a plain .pt checkpoint file.
    Handles: {"model_state_dict": ...} | {"model_state_dicts": [...]} | raw state dict
    """
    if is_main():
        print(f"  Format : plain .pt file")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unexpected checkpoint type: {type(ckpt)}. "
            "Expected a dict with 'model_state_dict', 'model_state_dicts', "
            "or a raw state dict."
        )

    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("model_state_dicts", [None])[0]
        or ckpt
    )

    dropped   = _drop_threshold_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if is_main():
        if dropped:    print(f"  Dropped {len(dropped)} threshold keys for fresh calibration.")
        if missing:    print(f"  Missing keys    : {len(missing)}")
        if unexpected: print(f"  Unexpected keys : {len(unexpected)}")

    return model


def load_pretrained(model, checkpoint_path: str):
    """Unified loader: dispatches to DCP or .pt based on whether the path is a directory."""
    path = Path(checkpoint_path)
    if is_main():
        print(f"Loading checkpoint: {checkpoint_path}")

    if path.is_dir():
        model = _load_from_dcp(model, checkpoint_path)
    elif path.is_file():
        model = _load_from_pt(model, checkpoint_path)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Provide either a TorchTitan DCP directory or a .pt file."
        )

    if is_main():
        print("Checkpoint loaded successfully.")

    return model


# ============================================================================
# Threshold calibration  (rank 0 only, then broadcast)
# ============================================================================

def _sample_rows(tensor, max_rows: int):
    if tensor is None or tensor.dim() < 2:
        return None
    rows = tensor.detach().float().cpu()
    if rows.dim() == 3:
        rows = rows.reshape(-1, rows.size(-1))
    if max_rows > 0 and rows.size(0) > max_rows:
        rows = rows[torch.randperm(rows.size(0))[:max_rows]]
    return rows


def _merge(existing, new_rows, max_rows: int):
    if new_rows is None:
        return existing
    merged = new_rows if existing is None else torch.cat([existing, new_rows], dim=0)
    if max_rows > 0 and merged.size(0) > max_rows:
        merged = merged[torch.randperm(merged.size(0))[:max_rows]]
    return merged


def _thresholds_from_std(samples, fallback, t_min: float, t_max: float, std_floor: float):
    if samples is None or samples.numel() == 0:
        return fallback.detach().clone().cpu(), {
            "samples": 0, "threshold_mean": float(fallback.mean()),
        }
    std       = samples.std(dim=0, unbiased=False).clamp_min(std_floor)
    threshold = (0.5 * std).clamp(t_min, t_max)
    return threshold, {
        "samples":        int(samples.size(0)),
        "std_mean":       float(std.mean()),
        "threshold_mean": float(threshold.mean()),
    }


@torch.no_grad()
def calibrate_thresholds(raw_model, calib_loader, device, args):
    """
    Run forward passes on `calib_loader` (rank 0 only) and set per-channel
    thresholds to 0.5 * std of the observed pre-quantisation activations.
    """
    was_training = raw_model.training
    raw_model.eval()

    input_fallbacks  = raw_model.get_layer_input_thresholds()
    output_fallbacks = raw_model.get_layer_output_thresholds()
    n_layers         = len(input_fallbacks)
    input_samples    = [None] * n_layers
    output_samples   = [None] * n_layers

    max_batches = _calibration_batch_limit(args)
    if max_batches <= 0:
        raise ValueError(
            "Calibration requires a positive --calibration_batches or "
            "--calibration_samples value."
        )

    seen = 0
    for batch_idx, batch in enumerate(calib_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        raw_model(input_ids=input_ids, apply_qat_quantization=False)

        for i, layer in enumerate(raw_model.backbone.layers):
            mixer = layer.mixer
            input_samples[i]  = _merge(
                input_samples[i],
                _sample_rows(mixer.last_input_prequant,  args.sample_rows_per_batch),
                args.max_rows_per_layer,
            )
            output_samples[i] = _merge(
                output_samples[i],
                _sample_rows(mixer.last_output_prequant, args.sample_rows_per_batch),
                args.max_rows_per_layer,
            )
        seen += 1

    if seen == 0:
        raise RuntimeError(
            "Calibration saw 0 batches — increase --calibration_batches or "
            "--calibration_samples."
        )

    in_thresholds, out_thresholds, report = [], [], []
    for i, (ir, or_, ib, ob) in enumerate(
        zip(input_samples, output_samples, input_fallbacks, output_fallbacks)
    ):
        it, is_ = _thresholds_from_std(ir, ib, args.threshold_min, args.threshold_max, args.std_floor)
        ot, os_ = _thresholds_from_std(or_, ob, args.threshold_min, args.threshold_max, args.std_floor)
        in_thresholds.append(it)
        out_thresholds.append(ot)
        report.append({"layer": i, "input": is_, "output": os_})

    raw_model.set_layer_input_thresholds(in_thresholds)
    raw_model.set_layer_output_thresholds(out_thresholds)

    if was_training:
        raw_model.train()
    return report


def broadcast_thresholds(raw_model):
    """Broadcast thresholds calibrated on rank 0 to all other ranks."""
    for layer in raw_model.backbone.layers:
        dist.broadcast(layer.mixer._input_threshold_raw.data,  src=0)
        dist.broadcast(layer.mixer._output_threshold_raw.data, src=0)


# ============================================================================
# QAT regularization: L1 on discrete temporal diffs
# ============================================================================

def _masked_mean_abs_diff(discrete):
    """Mean |x_t - x_{t-1}| on a (B, T, D) discrete tensor."""
    if discrete is None:
        return None
    prev = torch.cat([torch.zeros_like(discrete[:, :1]), discrete[:, :-1]], dim=1)
    return (discrete - prev).abs().mean()


def compute_qat_reg_loss(raw_model) -> torch.Tensor:
    """L1 of discrete temporal diffs, averaged across all layers."""
    per_layer = []
    for layer in raw_model.backbone.layers:
        mixer = layer.mixer
        terms = [
            t for t in [
                _masked_mean_abs_diff(mixer.last_input_discrete),
                _masked_mean_abs_diff(mixer.last_output_discrete),
            ]
            if t is not None
        ]
        if terms:
            per_layer.append(torch.stack(terms).mean())

    if not per_layer:
        return torch.zeros((), device=next(raw_model.parameters()).device)
    return torch.stack(per_layer).mean()


# ============================================================================
# Sparsity metrics — mirrors _add_sparsity_metrics from train.py
# ============================================================================

def _add_sparsity_metrics(extra_metrics: dict, output) -> None:
    sp = getattr(output, "activation_sparsities", None)
    if sp is None:
        return

    if isinstance(sp, dict):
        if sp.get("input")  is not None:
            extra_metrics["sparsity/act_input"]  = sp["input"]
        if sp.get("output") is not None:
            extra_metrics["sparsity/act_output"] = sp["output"]
    elif hasattr(sp, "__iter__"):
        sp_list = list(sp)
        if sp_list:
            extra_metrics["sparsity/act_overall"] = sum(sp_list) / len(sp_list)
            extra_metrics["sparsity/act_min"]     = min(sp_list)
            extra_metrics["sparsity/act_max"]     = max(sp_list)

    reg = getattr(output, "regularization_term", None)
    if reg is not None:
        extra_metrics["loss/reg_term"] = reg.item() if hasattr(reg, "item") else float(reg)
def build_optimizer(raw_model, args):
    threshold_params    = [p for p in raw_model.get_threshold_parameters() if p.requires_grad]
    threshold_param_ids = {id(p) for p in threshold_params}
    base_params = [
        p for p in raw_model.parameters()
        if p.requires_grad and id(p) not in threshold_param_ids
    ]

    param_groups = []
    if base_params:
        param_groups.append({
            "params":       base_params,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "group_name":   "base",
        })
    if threshold_params:
        param_groups.append({
            "params":       threshold_params,
            "lr":           args.threshold_lr,
            "weight_decay": 0.0,
            "group_name":   "thresholds",
        })

    optimizer = torch.optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), eps=args.eps,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    return optimizer, scheduler


def _threshold_group_indices(optimizer) -> list[int]:
    return [
        i for i, group in enumerate(optimizer.param_groups)
        if group.get("group_name") == "thresholds"
    ]


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(path, model, optimizer, scheduler, step, args, extra=None):
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "step":                 step,
        "model_state_dict":     raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args":                 vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ============================================================================
# WandB helpers
# ============================================================================

def maybe_init_wandb(args):
    if not args.enable_wandb or not HAS_WANDB or not is_main():
        return
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fla-lm-qat"),
        name=os.environ.get("WANDB_NAME",       "qat_lm_delta"),
        config=vars(args),
    )


def wandb_enabled():
    return HAS_WANDB and wandb.run is not None


# ============================================================================
# Main training loop
# ============================================================================

def finetune(args):
    local_rank, global_rank, world_size, device = setup_ddp()
    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        maybe_init_wandb(args)

        # ── Data ─────────────────────────────────────────────────────────────
        if is_main():
            print(
                f"Building dataloaders "
                f"(dataset={args.dataset}/{args.dataset_name}, "
                f"seq_len={args.seq_len}) ..."
            )
        train_loader, calib_loader, tokenizer = build_dataloaders(
            args, world_size, global_rank,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        config  = load_config(args.config_path, initial_threshold=args.initial_threshold)
        student = AutoModelForCausalLM.from_config(config).to(device)
        student = load_pretrained(student, args.checkpoint)

        # ── Calibrate thresholds on rank 0, then broadcast ───────────────────
        # Barrier here is intentional: all ranks must wait for calibration
        # before the broadcast that follows.
        calibration_report = []
        if is_main():
            if calib_loader is None:
                raise RuntimeError("calib_loader is None on rank 0 — this should not happen.")
            calibration_batches = _calibration_batch_limit(args)
            print(
                f"Calibrating thresholds on {args.calibration_samples} samples "
                f"({calibration_batches} batches max), k = 0.5 * std ..."
            )
            calibration_report = calibrate_thresholds(student, calib_loader, device, args)
            with open(output_dir / "calibration_report.json", "w") as f:
                json.dump(calibration_report, f, indent=2)
            for row in calibration_report:
                print(
                    f"  layer {row['layer']:02d}  "
                    f"in_k={row['input']['threshold_mean']:.5f}  "
                    f"out_k={row['output']['threshold_mean']:.5f}"
                )

        broadcast_thresholds(student)
        # Single barrier: synchronise after calibration + broadcast before any
        # forward pass with quantization enabled.
        dist.barrier()

        # ── Enable all params + thresholds for training ───────────────────────
        for p in student.parameters():
            p.requires_grad_(True)
        student.set_threshold_trainable(True)

        raw_student          = student
        optimizer, scheduler = build_optimizer(raw_student, args)
        threshold_group_idxs = _threshold_group_indices(optimizer)

        amp_dtype = (
            None if args.mixed_precision == "none"
            else getattr(torch, args.mixed_precision)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "float16"))

        student     = DDP(
            student, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        raw_student = student.module

        freeze_threshold_step = None
        thresholds_frozen = False
        if args.freeze_thresholds_at_80pct:
            freeze_threshold_step = max(1, math.ceil(0.8 * args.steps))
            if is_main():
                print(
                    "Threshold freezing enabled: input/output thresholds "
                    f"will be frozen at step {freeze_threshold_step}/{args.steps}."
                )

        # ── Training loop ─────────────────────────────────────────────────────
        train_iter = iter(train_loader)

        # Timing / throughput state — mirrors train.py's metric_logger fields
        ntokens_since_last_log = 0
        time_last_log          = time.perf_counter()
        elapsed                = timedelta()

        if is_main():
            print(f"Starting QAT finetuning for {args.steps} steps ...")

        for step in range(1, args.steps + 1):
            student.train()

            if (
                freeze_threshold_step is not None
                and not thresholds_frozen
                and step >= freeze_threshold_step
            ):
                for group_idx in threshold_group_idxs:
                    optimizer.param_groups[group_idx]["lr"] = 0.0
                    optimizer.param_groups[group_idx]["initial_lr"] = 0.0
                    scheduler.base_lrs[group_idx] = 0.0
                thresholds_frozen = True
                if is_main():
                    print(f"Freezing input/output thresholds at step {step}/{args.steps}.")

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # Track tokens for throughput logging
            ntokens_since_last_log += labels.numel()

            # Cosine ramp-up of the L1 sparsity weight
            l1_schedule_steps = (
                args.sparsity_l1_schedule_steps
                if args.sparsity_l1_schedule_steps is not None
                else args.warmup_steps
            )
            current_l1_weight = schedule_value(
                max(step - 1, 0), l1_schedule_steps,
                args.sparsity_l1_weight_start, args.sparsity_l1_weight,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                output      = student(
                    input_ids=input_ids,
                    labels=labels,
                    apply_qat_quantization=True,
                )
                task_loss   = output.loss
                sparsity_l1 = compute_qat_reg_loss(raw_student)
                loss        = args.lm_weight * task_loss + current_l1_weight * sparsity_l1

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.max_grad_norm,
            )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if thresholds_frozen:
                for group_idx in threshold_group_idxs:
                    optimizer.param_groups[group_idx]["lr"] = 0.0

            # ── Logging — mirrors train.py: all-reduce only at log frequency ─
            if step % args.log_freq == 0:
                # Single all-reduce for all three scalars at log time only
                stats = torch.tensor(
                    [task_loss.detach(), sparsity_l1.detach(), loss.detach()],
                    device=device, dtype=torch.float32,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
                mean_task, mean_l1, mean_loss = stats.tolist()

                grad_norm_tensor = torch.tensor(float(grad_norm), device=device)
                dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.AVG)

                if is_main():
                    time_now   = time.perf_counter()
                    time_delta = time_now - time_last_log
                    elapsed   += timedelta(seconds=time_delta)
                    time_last_log = time_now

                    eta = (
                        elapsed * (args.steps - step) / step
                        if step > 0 else timedelta()
                    )

                    sp = getattr(output, "activation_sparsities", None) or {}
                    last_lr = scheduler.get_last_lr()[0]

                    extra_metrics = {
                        "optimizer/lr":          last_lr,
                        "optimizer/grad_norm":   grad_norm_tensor.item(),
                        "train/task_loss":       mean_task,
                        "train/sparsity_l1":     mean_l1,
                        "train/loss":            mean_loss,
                        "train/l1_weight":       current_l1_weight,
                        "train/thresholds_frozen": float(thresholds_frozen),
                    }
                    _add_sparsity_metrics(extra_metrics, output)

                    print(
                        f"step={step:5d}/{args.steps}  "
                        f"task={mean_task:.4f}  l1={mean_l1:.6f}  loss={mean_loss:.4f}  "
                        f"l1_w={current_l1_weight:.4f}  "
                        f"lr={last_lr:.2e}  "
                        f"grad={grad_norm_tensor.item():.3f}  "
                        f"sp_in={sp.get('input')  or 0:.1f}%  "
                        f"sp_out={sp.get('output') or 0:.1f}%  "
                        f"[{str(elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]"
                    )

                    if wandb_enabled():
                        wandb.log(extra_metrics, step=step)

        # ── Final save — barrier ensures all ranks finish before rank 0 writes
        dist.barrier()
        if is_main():
            raw_s = student.module if isinstance(student, DDP) else student
            torch.save(raw_s.state_dict(), output_dir / "final_model.pt")
            torch.save(
                {
                    "args":                    vars(args),
                    "layer_input_thresholds":  [t.cpu() for t in raw_s.get_layer_input_thresholds()],
                    "layer_output_thresholds": [t.cpu() for t in raw_s.get_layer_output_thresholds()],
                    "calibration_report":      calibration_report,
                },
                output_dir / "final_thresholds.pt",
            )
            print("Done.")
            if wandb_enabled():
                wandb.finish()

        dist.barrier()

    finally:
        cleanup_ddp()


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="QAT finetuning for SparseMamba2DeltaForCausalLM"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",     required=True,
                   help="Pretrained checkpoint: TorchTitan DCP directory OR plain .pt file")
    p.add_argument("--config_path",    required=True,
                   help="Model config (json file or HF Hub directory)")
    p.add_argument("--output_dir",     default="./qat_lm_ckpts")
    p.add_argument("--tokenizer_path", default="fla-hub/transformer-1.3B-100B",
                   help="HF tokenizer path or Hub ID")

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",        default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name",   default="sample-100BT")
    p.add_argument("--dataset_split",  default="train")

    # ── Data / tokenisation ───────────────────────────────────────────────────
    p.add_argument("--seq_len",        type=int, default=2048)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--seed",           type=int, default=42)

    # ── Calibration ───────────────────────────────────────────────────────────
    p.add_argument("--initial_threshold",     type=float, default=None,
                   help="Override config threshold before calibration")
    p.add_argument("--calibration_samples",   type=int,   default=512,
                   help="Examples loaded eagerly on rank 0 for calibration")
    p.add_argument("--calibration_batches",   type=int,   default=8,
                   help="Max forward passes during calibration")
    p.add_argument("--sample_rows_per_batch", type=int,   default=256)
    p.add_argument("--max_rows_per_layer",    type=int,   default=8192)
    p.add_argument("--threshold_min",         type=float, default=1e-4)
    p.add_argument("--threshold_max",         type=float, default=1e4)
    p.add_argument("--std_floor",             type=float, default=1e-4)

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--steps",                       type=int,   default=5000)
    p.add_argument("--warmup_steps",                type=int,   default=500)
    p.add_argument("--lr",                          type=float, default=1e-5)
    p.add_argument("--threshold_lr",                type=float, default=5e-5)
    p.add_argument("--weight_decay",                type=float, default=0.01)
    p.add_argument("--beta1",                       type=float, default=0.9)
    p.add_argument("--beta2",                       type=float, default=0.95)
    p.add_argument("--eps",                         type=float, default=1e-8)
    p.add_argument("--lm_weight",                   type=float, default=1.0,
                   help="Weight for cross-entropy loss")
    p.add_argument("--sparsity_l1_weight",          type=float, default=0.01,
                   help="Final L1 weight on discrete temporal diffs")
    p.add_argument("--sparsity_l1_weight_start",    type=float, default=0.0,
                   help="Initial L1 weight (cosine ramp to --sparsity_l1_weight)")
    p.add_argument("--sparsity_l1_schedule_steps",  type=int,   default=None,
                   help="Steps over which to ramp L1 weight (default: warmup_steps)")
    p.add_argument("--max_grad_norm",               type=float, default=1.0)
    p.add_argument("--freeze_thresholds_at_80pct",  type=int,   default=0,
                   help="If set to 1, freeze input/output threshold params at 80% of --steps")
    p.add_argument("--mixed_precision",
                   choices=["none", "float16", "bfloat16"], default="bfloat16")

    # ── Logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--log_freq",       type=int, default=50)
    p.add_argument("--enable_wandb",   type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
