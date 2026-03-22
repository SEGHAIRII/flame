#!/usr/bin/env python3
"""
Teacher-student QAT finetuning for SparseMamba2DeltaASRForCTC with channel-wise thresholds.

Phase responsibilities
----------------------
- Pretraining  (train_parallel_asr.py) : no quantization, L2 reg on prequant input + output.
- QAT          (this file)             : quantization ON, L1 on discrete temporal differences.
- Evaluation   (eval_asr.py)           : quantization ON, discrete-diff sparsity + magnitude.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from flame.train_parallel_asr import calculate_asr_metrics, decode_predictions_ctc
from sparse_mamba.custom_dataloaders.preprocess_ctc import CachedASRDataset, collate_fn_ctc
from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
    SparseMamba2DeltaASRConfig,
    SparseMamba2DeltaASRForCTC,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# DDP HELPERS
# ============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, global_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# ============================================================================
# SCHEDULER
# ============================================================================

def schedule_value(step: int, total_steps: int, start_value: float, target_value: float) -> float:
    if total_steps <= 0 or step >= total_steps:
        return target_value
    progress = step / total_steps
    factor = 0.5 * (1.0 - math.cos(progress * math.pi))
    return start_value + (target_value - start_value) * factor


# ============================================================================
# WANDB
# ============================================================================

def maybe_init_wandb(args):
    if not args.enable_wandb or not HAS_WANDB or not is_main_process():
        return
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "fla-asr"),
        name=os.environ.get("WANDB_NAME", "qat_delta"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME"),
        config=vars(args),
    )


def wandb_enabled() -> bool:
    return HAS_WANDB and wandb.run is not None


# ============================================================================
# CONFIG / CHECKPOINT LOADING
# ============================================================================

def load_config(config_path: str, initial_threshold: Optional[float] = None) -> SparseMamba2DeltaASRConfig:
    config = SparseMamba2DeltaASRConfig.from_pretrained(config_path)
    if initial_threshold is not None:
        config.threshold = float(initial_threshold)
    # QAT phase: model provides no internal regularization;
    # sparsity is encouraged externally via L1 on discrete temporal diffs.
    config.regularization_mode = "none"
    return config


def _extract_state_dict(ckpt_obj):
    if "model_state_dicts" in ckpt_obj:
        return ckpt_obj["model_state_dicts"][0]
    if "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_pretrained(model: SparseMamba2DeltaASRForCTC, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    # Drop threshold keys so calibration can initialize them cleanly.
    dropped = [k for k in list(state_dict.keys()) if k.endswith("_threshold_raw")]
    for k in dropped:
        state_dict.pop(k, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if is_main_process():
        print(f"Loaded checkpoint from {checkpoint_path}")
        if dropped:
            print(f"  Dropped {len(dropped)} threshold keys for clean calibration.")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    return model


def _strip_wrapper_prefixes(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def _map_mamba2_noconv_key_to_delta(key: str) -> str:
    key = _strip_wrapper_prefixes(key)
    if key.startswith("sparse_mamba_asr."):
        return "sparse_mamba_delta_asr." + key[len("sparse_mamba_asr."):]
    if key.startswith("sparse_mamba2_asr_noconv."):
        return "sparse_mamba_delta_asr." + key[len("sparse_mamba2_asr_noconv."):]
    return key


def load_mamba2_noconv_into_delta(model: SparseMamba2DeltaASRForCTC, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state_dict = _extract_state_dict(ckpt)
    target_state_dict = model.state_dict()

    remapped_state_dict = {}
    skipped_missing_target = []
    skipped_shape_mismatch = []

    for src_key, src_value in source_state_dict.items():
        if not torch.is_tensor(src_value):
            continue

        dst_key = _map_mamba2_noconv_key_to_delta(src_key)
        dst_value = target_state_dict.get(dst_key)
        if dst_value is None:
            skipped_missing_target.append((src_key, dst_key))
            continue
        if src_value.shape != dst_value.shape:
            skipped_shape_mismatch.append(
                (src_key, dst_key, tuple(src_value.shape), tuple(dst_value.shape))
            )
            continue
        remapped_state_dict[dst_key] = src_value

    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)

    if is_main_process():
        print(f"Initialized Delta model from Mamba2-no-conv checkpoint: {checkpoint_path}")
        print(f"  Loaded compatible tensors: {len(remapped_state_dict)}")
        print(f"  Source keys without Delta target: {len(skipped_missing_target)}")
        print(f"  Source keys skipped due to shape mismatch: {len(skipped_shape_mismatch)}")
        if missing:
            print(f"  Delta keys not loaded (expected for Delta-only params): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    return model


# ============================================================================
# DATALOADERS
# ============================================================================

def build_dataloaders(args, world_size: int, global_rank: int):
    if args.dataset == "librispeech":
        train_split, val_split, test_split = "train.960", "validation", "test"
    else:
        train_split, val_split, test_split = "train", "validation", "test"

    train_ds = CachedASRDataset(args.cache_dir, train_split, args.max_samples)
    val_ds  = CachedASRDataset(args.cache_dir, val_split,  args.max_samples) if global_rank == 0 else None
    test_ds = CachedASRDataset(args.cache_dir, test_split, args.max_samples) if global_rank == 0 else None

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn_ctc, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds if val_ds is not None else train_ds,
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_ctc, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds if test_ds is not None else train_ds,
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_ctc, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    calib_loader = None
    if global_rank == 0:
        if args.calibration_source == "validation":
            calib_loader = val_loader
        else:
            n_train = len(train_ds)
            n_calib = args.calibration_train_samples
            if n_calib <= 0:
                n_calib = min(n_train, args.calibration_batches * args.batch_size)
            n_calib = min(n_train, n_calib)
            generator = torch.Generator()
            generator.manual_seed(args.calibration_seed)
            indices = torch.randperm(n_train, generator=generator)[:n_calib].tolist()
            calib_loader = torch.utils.data.DataLoader(
                Subset(train_ds, indices),
                batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn_ctc, num_workers=args.num_workers,
                pin_memory=True, drop_last=False,
            )

    sample_features = train_ds[0]["features"]
    input_dim = sample_features.size(0)

    for loader in [train_loader, val_loader, test_loader] + ([calib_loader] if calib_loader else []):
        loader.n_classes   = len(train_ds.char_to_idx)
        loader.input_dim   = input_dim
        loader.seq_len     = -1
        loader.char_to_idx = train_ds.char_to_idx
        loader.idx_to_char = train_ds.idx_to_char

    return train_loader, val_loader, test_loader, calib_loader, train_sampler


# ============================================================================
# CALIBRATION HELPERS
# ============================================================================

def _compute_output_lengths(raw_model: SparseMamba2DeltaASRForCTC, input_lengths: torch.Tensor) -> torch.Tensor:
    encoder = getattr(raw_model, "sparse_mamba_delta_asr", None)
    if encoder is None or getattr(encoder, "conv_subsampler", None) is None:
        return input_lengths
    kernel  = int(getattr(raw_model.config, "conv_subsample_kernel", 3))
    stride  = int(getattr(raw_model.config, "conv_subsample_stride", 2))
    padding = kernel // 2
    out = encoder._conv1d_out_lengths(input_lengths, kernel, stride, padding)
    return encoder._conv1d_out_lengths(out, kernel, stride, padding)


def _sample_valid_rows(tensor, output_lengths, max_rows):
    if tensor is None:
        return None
    batch_size, seq_len, _ = tensor.shape
    positions = torch.arange(seq_len, device=tensor.device).unsqueeze(0)
    valid = positions < output_lengths.to(tensor.device).unsqueeze(1)
    rows = tensor[valid].detach().float().cpu()
    if rows.numel() == 0:
        return None
    if max_rows > 0 and rows.size(0) > max_rows:
        rows = rows[torch.randperm(rows.size(0))[:max_rows]]
    return rows


def _merge_row_samples(existing, new_rows, max_rows):
    if new_rows is None:
        return existing
    merged = new_rows if existing is None else torch.cat([existing, new_rows], dim=0)
    if max_rows > 0 and merged.size(0) > max_rows:
        merged = merged[torch.randperm(merged.size(0))[:max_rows]]
    return merged


def _thresholds_from_std_samples(samples, fallback, threshold_min, threshold_max, std_floor):
    if samples is None or samples.numel() == 0:
        threshold = fallback.detach().clone().cpu()
        return threshold, {
            "samples": 0, "std_min": 0.0, "std_mean": 0.0, "std_max": 0.0,
            "threshold_min": float(threshold.min()), "threshold_mean": float(threshold.mean()),
            "threshold_max": float(threshold.max()),
        }
    std = samples.std(dim=0, unbiased=False).clamp_min(std_floor)
    threshold = (0.5 * std).clamp(threshold_min, threshold_max)
    return threshold, {
        "samples": int(samples.size(0)),
        "std_min":  float(std.min()),  "std_mean":  float(std.mean()),  "std_max":  float(std.max()),
        "threshold_min": float(threshold.min()), "threshold_mean": float(threshold.mean()),
        "threshold_max": float(threshold.max()),
    }


@torch.no_grad()
def calibrate_thresholds(raw_model: SparseMamba2DeltaASRForCTC, calib_loader, device, args):
    was_training = raw_model.training
    raw_model.eval()

    input_fallbacks  = raw_model.get_layer_input_thresholds()
    output_fallbacks = raw_model.get_layer_output_thresholds()
    input_samples  = [None] * len(input_fallbacks)
    output_samples = [None] * len(output_fallbacks)

    seen = 0
    for batch_idx, batch in enumerate(calib_loader):
        if batch_idx >= args.calibration_batches:
            break
        inputs        = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)

        _ = raw_model(input_ids=inputs, input_lengths=input_lengths, apply_qat_quantization=False)
        output_lengths = _compute_output_lengths(raw_model, input_lengths)

        for layer_idx, layer in enumerate(raw_model.sparse_mamba_delta_asr.layers):
            mixer = layer.mixer
            input_samples[layer_idx] = _merge_row_samples(
                input_samples[layer_idx],
                _sample_valid_rows(mixer.last_input_prequant, output_lengths, args.sample_rows_per_batch),
                args.max_rows_per_layer,
            )
            output_samples[layer_idx] = _merge_row_samples(
                output_samples[layer_idx],
                _sample_valid_rows(mixer.last_output_prequant, output_lengths, args.sample_rows_per_batch),
                args.max_rows_per_layer,
            )
        seen += 1

    if seen == 0:
        raise RuntimeError("Calibration saw zero batches. Increase --calibration_batches.")

    layer_input_thresholds, layer_output_thresholds, report = [], [], []
    for layer_idx, (ir, or_, ib, ob) in enumerate(
        zip(input_samples, output_samples, input_fallbacks, output_fallbacks)
    ):
        it, is_ = _thresholds_from_std_samples(ir, ib, args.threshold_min, args.threshold_max, args.std_floor)
        ot, os_ = _thresholds_from_std_samples(or_, ob, args.threshold_min, args.threshold_max, args.std_floor)
        layer_input_thresholds.append(it)
        layer_output_thresholds.append(ot)
        report.append({"layer": layer_idx, "input": is_, "output": os_})

    raw_model.set_layer_input_thresholds(layer_input_thresholds)
    raw_model.set_layer_output_thresholds(layer_output_thresholds)

    if was_training:
        raw_model.train()
    return report


def broadcast_thresholds(raw_model: SparseMamba2DeltaASRForCTC):
    for layer in raw_model.sparse_mamba_delta_asr.layers:
        dist.broadcast(layer.mixer._input_threshold_raw.data, src=0)
        dist.broadcast(layer.mixer._output_threshold_raw.data, src=0)


# ============================================================================
# DISCRETE TEMPORAL DIFF METRICS  (used in both QAT eval and final eval)
# ============================================================================


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device, max_batches: int = 30, apply_qat_quantization: bool = True):
    raw_model = model.module if isinstance(model, DDP) else model
    was_training = raw_model.training
    raw_model.eval()

    total_ctc, total_reg        = 0.0, 0.0
    sparsity_input_acc          = []
    sparsity_output_acc         = []
    all_preds, all_targets      = [], []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        inputs         = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch["targets"].to(device, dtype=torch.long)
        input_lengths  = batch["feature_lengths"].to(device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

        output = raw_model(
            input_ids=inputs, labels=labels,
            input_lengths=input_lengths, target_lengths=target_lengths,
            apply_qat_quantization=apply_qat_quantization,
        )

        if output.loss is not None:
            total_ctc += float(output.loss.item())
        if output.regularization_term is not None:
            total_reg += float(output.regularization_term.item())

        sp = getattr(output, 'all_sparsities', None)
        if isinstance(sp, dict):
            if sp.get('input')  is not None: sparsity_input_acc.append(sp['input'])
            if sp.get('output') is not None: sparsity_output_acc.append(sp['output'])

        output_lengths = (
            output.output_lengths if output.output_lengths is not None
            else _compute_output_lengths(raw_model, input_lengths)
        )
        preds = decode_predictions_ctc(output.logits, output_lengths, blank_id=raw_model.blank_id)
        all_preds.extend(preds)
        start = 0
        for sample_idx in range(len(preds)):
            tl = target_lengths[sample_idx].item()
            all_targets.append(labels[start:start + tl].tolist())
            start += tl

    num_batches = max(1, min(len(dataloader), max_batches if max_batches > 0 else len(dataloader)))
    metrics = calculate_asr_metrics(all_preds, all_targets, dataloader.idx_to_char)

    if was_training:
        raw_model.train()
    else:
        raw_model.eval()

    return {
        "ctc":            total_ctc / num_batches,
        "reg":            total_reg / num_batches,
        "cer":            metrics["cer"],
        "wer":            metrics["wer"],
        "sparsity_input":  sum(sparsity_input_acc)  / len(sparsity_input_acc)  if sparsity_input_acc  else None,
        "sparsity_output": sum(sparsity_output_acc) / len(sparsity_output_acc) if sparsity_output_acc else None,
    }


# ============================================================================
# REGULARIZATION LOSS  (QAT phase: L1 on discrete temporal diffs)
# ============================================================================

def _valid_token_mask(output_lengths, max_len, device):
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < output_lengths.to(device).unsqueeze(1)


def _masked_mean_abs_diff(discrete, output_lengths):
    """Mean absolute value of temporal differences of discrete activations."""
    if discrete is None:
        return None
    mask = _valid_token_mask(output_lengths, discrete.size(1), discrete.device).to(discrete.dtype)
    while mask.dim() < discrete.dim():
        mask = mask.unsqueeze(-1)
    prev = torch.cat([torch.zeros_like(discrete[:, :1]), discrete[:, :-1]], dim=1)
    diff = (discrete - prev).abs()
    denom = mask.sum().clamp_min(1.0) * discrete.size(-1)
    return (diff * mask).sum() / denom


def compute_regularization_losses(raw_model: SparseMamba2DeltaASRForCTC, output_lengths: torch.Tensor):
    """
    L1 sparsity loss on discrete temporal differences (QAT phase only).
    Encourages consecutive quantized values to stay the same (zero diff = free update).
    """
    per_layer = []
    for layer in raw_model.sparse_mamba_delta_asr.layers:
        mixer = layer.mixer
        terms = []
        for discrete in (mixer.last_input_discrete, mixer.last_output_discrete):
            term = _masked_mean_abs_diff(discrete, output_lengths)
            if term is not None:
                terms.append(term)
        if terms:
            per_layer.append(torch.stack(terms).mean())

    device = output_lengths.device
    return torch.stack(per_layer).mean() if per_layer else torch.zeros((), device=device)


# ============================================================================
# OPTIMIZER
# ============================================================================

def build_optimizer(raw_model: SparseMamba2DeltaASRForCTC, args):
    threshold_params    = [p for p in raw_model.get_threshold_parameters() if p.requires_grad]
    threshold_param_ids = {id(p) for p in threshold_params}
    base_params = [
        p for p in raw_model.parameters()
        if p.requires_grad and id(p) not in threshold_param_ids
    ]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if threshold_params:
        param_groups.append({"params": threshold_params, "lr": args.threshold_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), eps=args.eps,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.steps,
    )
    return optimizer, scheduler


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(path, model, optimizer, scheduler, step, args, extra):
    raw_model = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    payload.update(extra)
    torch.save(payload, path)


# ============================================================================
# MAIN FINETUNE LOOP
# ============================================================================

def finetune(args):
    local_rank, global_rank, world_size, device = setup_ddp()
    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.enable_wandb and not HAS_WANDB and is_main_process():
            print("WandB requested but not installed — continuing without it.")
        maybe_init_wandb(args)

        train_loader, val_loader, test_loader, calib_loader, train_sampler = \
            build_dataloaders(args, world_size, global_rank)

        config  = load_config(args.config_path, initial_threshold=args.initial_threshold)
        student = SparseMamba2DeltaASRForCTC(config).to(device)
        if args.init_from_mamba2_noconv:
            student = load_mamba2_noconv_into_delta(student, args.checkpoint)
        else:
            student = load_pretrained(student, args.checkpoint)

        baseline_metrics      = None
        student_init_metrics  = None
        calibration_report    = []

        if is_main_process():
            # Baseline: no quantization
            print("Evaluating pretrained baseline (quantization disabled)...")
            baseline_metrics = evaluate_model(
                student, val_loader, device,
                max_batches=args.eval_batches, apply_qat_quantization=False,
            )
            print(
                f"  baseline  CTC={baseline_metrics['ctc']:.4f} "
                f"CER={baseline_metrics['cer']:.4f} WER={baseline_metrics['wer']:.4f}"
            )
            if wandb_enabled():
                wandb.log({
                    "baseline/val_ctc":          baseline_metrics["ctc"],
                    "baseline/val_cer":          baseline_metrics["cer"],
                    "baseline/val_wer":          baseline_metrics["wer"],
                    "baseline/sparsity_input":   baseline_metrics.get("sparsity_input",  0) or 0,
                    "baseline/sparsity_output":  baseline_metrics.get("sparsity_output", 0) or 0,
                }, step=0)

            if calib_loader is None:
                raise RuntimeError("Calibration loader is not available on rank 0.")

            print(f"Running calibration (k = 0.5 * std per channel)...")
            calibration_report = calibrate_thresholds(student, calib_loader, device, args)
            with open(output_dir / "calibration_report.json", "w") as f:
                json.dump(calibration_report, f, indent=2)
            torch.save({
                "args": vars(args),
                "calibration_report": calibration_report,
                "layer_input_thresholds":  [t.cpu() for t in student.get_layer_input_thresholds()],
                "layer_output_thresholds": [t.cpu() for t in student.get_layer_output_thresholds()],
            }, output_dir / "calibrated_thresholds.pt")

            for row in calibration_report:
                print(
                    f"  layer {row['layer']:02d} | "
                    f"in_std={row['input']['std_mean']:.5f}  "
                    f"out_std={row['output']['std_mean']:.5f}  "
                    f"in_k={row['input']['threshold_mean']:.5f}  "
                    f"out_k={row['output']['threshold_mean']:.5f}"
                )

            # Student after calibration, quantization ON
            print("Evaluating calibrated quantized student...")
            student_init_metrics = evaluate_model(
                student, val_loader, device,
                max_batches=args.eval_batches, apply_qat_quantization=True,
            )
            print(
                f"  student   CTC={student_init_metrics['ctc']:.4f} "
                f"CER={student_init_metrics['cer']:.4f} "
                f"WER={student_init_metrics['wer']:.4f} "
                f"sp_in={student_init_metrics.get('sparsity_input') or 0:.1f}% "
                f"sp_out={student_init_metrics.get('sparsity_output') or 0:.1f}%"
            )
            if wandb_enabled():
                wandb.log({
                    "student_init/val_ctc":        student_init_metrics["ctc"],
                    "student_init/val_cer":        student_init_metrics["cer"],
                    "student_init/val_wer":        student_init_metrics["wer"],
                    "student_init/sparsity_input":  student_init_metrics.get("sparsity_input", 0) or 0,
                    "student_init/sparsity_output": student_init_metrics.get("sparsity_output", 0) or 0,
                }, step=0)

        broadcast_thresholds(student)
        dist.barrier()

        for param in student.parameters():
            param.requires_grad_(True)
        student.set_threshold_trainable(True)

        raw_student = student
        optimizer, scheduler = build_optimizer(raw_student, args)
        amp_dtype = None if args.mixed_precision == "none" else getattr(torch, args.mixed_precision)
        scaler    = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "float16")

        student     = DDP(student, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False)
        raw_student = student.module

        best_cer = float("inf")
        train_sampler.set_epoch(0)
        train_iter = iter(train_loader)

        if is_main_process():
            print(f"Starting QAT finetuning for {args.steps} steps...")

        for step in range(1, args.steps + 1):
            student.train()
            raw_student.sparse_mamba_delta_asr.spec_augment.eval()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_sampler.set_epoch(step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            inputs         = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
            labels         = batch["targets"].to(device, dtype=torch.long)
            input_lengths  = batch["feature_lengths"].to(device, dtype=torch.long)
            target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            l1_schedule_steps = (
                args.sparsity_l1_schedule_steps
                if args.sparsity_l1_schedule_steps is not None
                else args.warmup_steps
            )
            current_l1_weight = schedule_value(
                max(step - 1, 0), l1_schedule_steps,
                args.sparsity_l1_weight_start, args.sparsity_l1_weight,
            )

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                student_output = student(
                    input_ids=inputs, labels=labels,
                    input_lengths=input_lengths, target_lengths=target_lengths,
                    apply_qat_quantization=True,
                )
                output_lengths = (
                    student_output.output_lengths
                    if student_output.output_lengths is not None
                    else _compute_output_lengths(raw_student, input_lengths)
                )
                task_loss  = student_output.loss
                if task_loss is None:
                    raise RuntimeError("Student output did not provide a task loss.")

                # QAT regularization: L1 on discrete temporal differences
                sparsity_l1 = compute_regularization_losses(raw_student, output_lengths)
                loss = args.ctc_weight * task_loss + current_l1_weight * sparsity_l1

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            # Aggregate across GPUs
            stats = torch.tensor(
                [task_loss.detach(), sparsity_l1.detach(), loss.detach()],
                device=device, dtype=torch.float32,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            mean_task, mean_l1, mean_loss = [x.item() for x in stats]

            if step % args.log_freq == 0 and is_main_process():
                print(
                    f"step={step:5d}/{args.steps} "
                    f"task={mean_task:.4f}  sparsity_l1={mean_l1:.6f}  "
                    f"loss={mean_loss:.4f}  l1_w={current_l1_weight:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  grad={float(grad_norm):.3f}"
                )
                if wandb_enabled():
                    wandb.log({
                        "train/task":            mean_task,
                        "train/sparsity_l1":     mean_l1,
                        "train/loss":            mean_loss,
                        "train/l1_weight":       current_l1_weight,
                        "train/lr":              scheduler.get_last_lr()[0],
                        "train/grad_norm":       float(grad_norm),
                    }, step=step)

            dist.barrier()

            if step % args.val_freq == 0 and is_main_process():
                val_metrics = evaluate_model(
                    student, val_loader, device,
                    max_batches=args.eval_batches, apply_qat_quantization=True,
                )
                print(
                    f"  [val] step={step}  CTC={val_metrics['ctc']:.4f}  "
                    f"CER={val_metrics['cer']:.4f}  WER={val_metrics['wer']:.4f}  "
                    f"sp_in={val_metrics.get('sparsity_input') or 0:.1f}%  "
                    f"sp_out={val_metrics.get('sparsity_output') or 0:.1f}%"
                )
                if wandb_enabled():
                    wandb.log({
                        "val/ctc":              val_metrics["ctc"],
                        "val/cer":              val_metrics["cer"],
                        "val/wer":              val_metrics["wer"],
                        "val/sparsity_input":   val_metrics.get("sparsity_input",  0) or 0,
                        "val/sparsity_output":  val_metrics.get("sparsity_output", 0) or 0,
                    }, step=step)

                save_checkpoint(output_dir / "latest.pt", student, optimizer, scheduler, step, args, {
                    "val_metrics": val_metrics,
                    "baseline_metrics": baseline_metrics,
                    "student_init_metrics": student_init_metrics,
                    "calibration_report": calibration_report,
                })

                if val_metrics["cer"] < best_cer:
                    best_cer = val_metrics["cer"]
                    save_checkpoint(output_dir / "best.pt", student, optimizer, scheduler, step, args, {
                        "val_metrics": val_metrics,
                        "baseline_metrics": baseline_metrics,
                        "student_init_metrics": student_init_metrics,
                        "calibration_report": calibration_report,
                    })
                    print(f"  New best checkpoint (CER={best_cer:.4f})")

            dist.barrier()

        dist.barrier()
        if is_main_process():
            print("Running final evaluations...")
            final_val  = evaluate_model(student, val_loader,  device,
                                        max_batches=max(args.eval_batches, 50), apply_qat_quantization=True)
            final_test = evaluate_model(student, test_loader, device,
                                        max_batches=max(args.eval_batches, 50), apply_qat_quantization=True)

            print(
                f"  final val   CTC={final_val['ctc']:.4f}  CER={final_val['cer']:.4f}  "
                f"WER={final_val['wer']:.4f}  "
                f"sp_in={final_val.get('sparsity_input') or 0:.1f}%  sp_out={final_val.get('sparsity_output') or 0:.1f}%"
            )
            print(
                f"  final test  CTC={final_test['ctc']:.4f}  CER={final_test['cer']:.4f}  "
                f"WER={final_test['wer']:.4f}  "
                f"sp_in={final_test.get('sparsity_input') or 0:.1f}%  sp_out={final_test.get('sparsity_output') or 0:.1f}%"
            )

            raw_s = student.module if isinstance(student, DDP) else student
            torch.save(raw_s.state_dict(), output_dir / "final_model.pt")
            torch.save({
                "args": vars(args),
                "layer_input_thresholds":  [t.cpu() for t in raw_s.get_layer_input_thresholds()],
                "layer_output_thresholds": [t.cpu() for t in raw_s.get_layer_output_thresholds()],
                "baseline_metrics":       baseline_metrics,
                "student_init_metrics":   student_init_metrics,
                "calibration_report":     calibration_report,
                "final_val_metrics":      final_val,
                "final_test_metrics":     final_test,
            }, output_dir / "final_thresholds.pt")

            if wandb_enabled():
                wandb.log({
                    "final/val_ctc":           final_val["ctc"],
                    "final/val_cer":           final_val["cer"],
                    "final/val_wer":           final_val["wer"],
                    "final/val_sparsity_input":  final_val.get("sparsity_input",  0) or 0,
                    "final/val_sparsity_output": final_val.get("sparsity_output", 0) or 0,
                    "final/test_ctc":          final_test["ctc"],
                    "final/test_cer":          final_test["cer"],
                    "final/test_wer":          final_test["wer"],
                    "final/test_sparsity_input":  final_test.get("sparsity_input",  0) or 0,
                    "final/test_sparsity_output": final_test.get("sparsity_output", 0) or 0,
                }, step=args.steps)

        dist.barrier()

    finally:
        if wandb_enabled() and is_main_process():
            wandb.finish()
        cleanup_ddp()


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config_path",  required=True)
    p.add_argument("--output_dir",   default="./qat_delta_ckpts")
    p.add_argument("--cache_dir",    required=True)
    p.add_argument(
        "--init_from_mamba2_noconv",
        action="store_true",
        help="Treat --checkpoint as a SparseMamba2-no-conv checkpoint and "
             "initialize the Delta model with all compatible weights.",
    )

    p.add_argument("--dataset",      default="librispeech")
    p.add_argument("--max_samples",  type=int, default=-1)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--seed",         type=int, default=1234)

    p.add_argument("--initial_threshold",          type=float, default=None)
    p.add_argument("--calibration_source",         choices=["validation", "train_subset"], default="train_subset")
    p.add_argument("--calibration_batches",        type=int,   default=64)
    p.add_argument("--calibration_train_samples",  type=int,   default=16000)
    p.add_argument("--calibration_seed",           type=int,   default=1234)
    p.add_argument("--sample_rows_per_batch",      type=int,   default=128)
    p.add_argument("--max_rows_per_layer",         type=int,   default=4096)
    p.add_argument("--threshold_min",              type=float, default=1e-4)
    p.add_argument("--threshold_max",              type=float, default=1e4)
    p.add_argument("--std_floor",                  type=float, default=1e-4)

    p.add_argument("--steps",                      type=int,   default=2000)
    p.add_argument("--warmup_steps",               type=int,   default=200)
    p.add_argument("--lr",                         type=float, default=1e-5)
    p.add_argument("--threshold_lr",               type=float, default=5e-5)
    p.add_argument("--weight_decay",               type=float, default=0.01)
    p.add_argument("--beta1",                      type=float, default=0.9)
    p.add_argument("--beta2",                      type=float, default=0.95)
    p.add_argument("--eps",                        type=float, default=1e-8)
    p.add_argument("--ctc_weight",                 type=float, default=1.0)
    p.add_argument("--sparsity_l1_weight",         type=float, default=0.05)
    p.add_argument("--sparsity_l1_weight_start",   type=float, default=0.0)
    p.add_argument("--sparsity_l1_schedule_steps", type=int,   default=None)
    p.add_argument("--max_grad_norm",              type=float, default=1.0)
    p.add_argument("--mixed_precision",            choices=["none", "float16", "bfloat16"], default="bfloat16")

    p.add_argument("--log_freq",    type=int, default=50)
    p.add_argument("--val_freq",    type=int, default=500)
    p.add_argument("--eval_batches",type=int, default=30)
    p.add_argument("--enable_wandb",type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
