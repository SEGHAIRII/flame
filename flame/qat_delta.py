#!/usr/bin/env python3
"""
Teacher-student QAT finetuning for SparseMamba2DeltaASRForCTC with channel-wise thresholds.
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


def schedule_value(step: int, total_steps: int, start_value: float, target_value: float) -> float:
    if total_steps <= 0 or step >= total_steps:
        return target_value
    progress = step / total_steps
    factor = 0.5 * (1.0 - math.cos(progress * math.pi))
    return start_value + (target_value - start_value) * factor


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


def _log_thresholds_wandb(raw_model: SparseMamba2DeltaASRForCTC, step: int, prefix: str = "thresholds"):
    if not wandb_enabled():
        return

    input_thresholds = raw_model.get_layer_input_thresholds()
    output_thresholds = raw_model.get_layer_output_thresholds()
    log_data = {}

    for kind, thresholds in (("input", input_thresholds), ("output", output_thresholds)):
        if not thresholds:
            continue
        flat = torch.cat([t.detach().cpu().float() for t in thresholds], dim=0)
        base_key = f"{prefix}/{kind}"
        log_data[f"{base_key}/global_mean"] = float(flat.mean().item())
        log_data[f"{base_key}/global_min"] = float(flat.min().item())
        log_data[f"{base_key}/global_max"] = float(flat.max().item())
        log_data[f"{base_key}/global_hist"] = wandb.Histogram(flat.numpy())
        for layer_idx, threshold in enumerate(thresholds):
            tt = threshold.detach().cpu().float()
            layer_key = f"{base_key}/layer_{layer_idx:02d}"
            log_data[f"{layer_key}/mean"] = float(tt.mean().item())
            log_data[f"{layer_key}/min"] = float(tt.min().item())
            log_data[f"{layer_key}/max"] = float(tt.max().item())

    if log_data:
        wandb.log(log_data, step=step)


def _log_calibration_report_wandb(calibration_report, step: int):
    if not wandb_enabled() or not calibration_report:
        return

    table = wandb.Table(
        columns=[
            "layer",
            "input_samples",
            "input_std_mean",
            "input_threshold_mean",
            "output_samples",
            "output_std_mean",
            "output_threshold_mean",
        ]
    )

    input_std = []
    output_std = []
    input_tau = []
    output_tau = []
    for row in calibration_report:
        input_stats = row["input"]
        output_stats = row["output"]
        input_std.append(float(input_stats["std_mean"]))
        output_std.append(float(output_stats["std_mean"]))
        input_tau.append(float(input_stats["threshold_mean"]))
        output_tau.append(float(output_stats["threshold_mean"]))
        table.add_data(
            int(row["layer"]),
            int(input_stats["samples"]),
            float(input_stats["std_mean"]),
            float(input_stats["threshold_mean"]),
            int(output_stats["samples"]),
            float(output_stats["std_mean"]),
            float(output_stats["threshold_mean"]),
        )

    wandb.log(
        {
            "calibration/input_std_mean": sum(input_std) / len(input_std),
            "calibration/output_std_mean": sum(output_std) / len(output_std),
            "calibration/input_threshold_mean": sum(input_tau) / len(input_tau),
            "calibration/output_threshold_mean": sum(output_tau) / len(output_tau),
            "calibration/report_table": table,
        },
        step=step,
    )


def load_config(config_path: str, initial_threshold: Optional[float] = None) -> SparseMamba2DeltaASRConfig:
    config = SparseMamba2DeltaASRConfig.from_pretrained(config_path)
    if initial_threshold is not None:
        config.threshold = float(initial_threshold)
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

    dropped = [key for key in list(state_dict.keys()) if key.endswith("_threshold_raw")]
    for key in dropped:
        state_dict.pop(key, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if is_main_process():
        print(f"Loaded checkpoint from {checkpoint_path}")
        if dropped:
            print(f"  Dropped {len(dropped)} threshold keys so calibration can initialize them cleanly.")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    return model


def build_dataloaders(args, world_size: int, global_rank: int):
    if args.dataset == "librispeech":
        train_split, val_split, test_split = "train.960", "validation", "test"
    else:
        train_split, val_split, test_split = "train", "validation", "test"

    train_ds = CachedASRDataset(args.cache_dir, train_split, args.max_samples)
    val_ds = CachedASRDataset(args.cache_dir, val_split, args.max_samples) if global_rank == 0 else None
    test_ds = CachedASRDataset(args.cache_dir, test_split, args.max_samples) if global_rank == 0 else None

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn_ctc,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds if val_ds is not None else train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds if test_ds is not None else train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
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
            calib_ds = Subset(train_ds, indices)
            calib_loader = torch.utils.data.DataLoader(
                calib_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn_ctc,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    sample_features = train_ds[0]["features"]
    input_dim = sample_features.size(0)
    loaders_to_annotate = [train_loader, val_loader, test_loader]
    if calib_loader is not None:
        loaders_to_annotate.append(calib_loader)

    for loader in loaders_to_annotate:
        loader.n_classes = len(train_ds.char_to_idx)
        loader.input_dim = input_dim
        loader.seq_len = -1
        loader.char_to_idx = train_ds.char_to_idx
        loader.idx_to_char = train_ds.idx_to_char

    return train_loader, val_loader, test_loader, calib_loader, train_sampler


def _compute_output_lengths(raw_model: SparseMamba2DeltaASRForCTC, input_lengths: torch.Tensor) -> torch.Tensor:
    output_lengths = input_lengths
    encoder = getattr(raw_model, "sparse_mamba_delta_asr", None)
    if encoder is None or getattr(encoder, "conv_subsampler", None) is None:
        return output_lengths

    kernel = int(getattr(raw_model.config, "conv_subsample_kernel", 3))
    stride = int(getattr(raw_model.config, "conv_subsample_stride", 2))
    padding = kernel // 2
    output_lengths = encoder._conv1d_out_lengths(output_lengths, kernel, stride, padding)
    output_lengths = encoder._conv1d_out_lengths(output_lengths, kernel, stride, padding)
    return output_lengths


def _sample_valid_rows(
    tensor: Optional[torch.Tensor],
    output_lengths: torch.Tensor,
    max_rows: int,
    take_abs: bool = False,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    batch_size, seq_len, _ = tensor.shape
    positions = torch.arange(seq_len, device=tensor.device).unsqueeze(0)
    valid = positions < output_lengths.to(tensor.device).unsqueeze(1)
    rows = tensor[valid]
    if rows.numel() == 0:
        return None
    rows = rows.detach().float().cpu()
    if take_abs:
        rows = rows.abs()
    if max_rows > 0 and rows.size(0) > max_rows:
        idx = torch.randperm(rows.size(0))[:max_rows]
        rows = rows[idx]
    return rows


def _merge_row_samples(
    existing: Optional[torch.Tensor],
    new_rows: Optional[torch.Tensor],
    max_rows: int,
) -> Optional[torch.Tensor]:
    if new_rows is None:
        return existing
    if existing is None:
        merged = new_rows
    else:
        merged = torch.cat([existing, new_rows], dim=0)
    if max_rows > 0 and merged.size(0) > max_rows:
        idx = torch.randperm(merged.size(0))[:max_rows]
        merged = merged[idx]
    return merged


def _thresholds_from_std_samples(
    samples: Optional[torch.Tensor],
    fallback: torch.Tensor,
    threshold_min: float,
    threshold_max: float,
    std_floor: float,
) -> Tuple[torch.Tensor, dict]:
    if samples is None or samples.numel() == 0:
        threshold = fallback.detach().clone().cpu()
        return threshold, {
            "samples": 0,
            "std_min": 0.0,
            "std_mean": 0.0,
            "std_max": 0.0,
            "threshold_min": float(threshold.min().item()),
            "threshold_mean": float(threshold.mean().item()),
            "threshold_max": float(threshold.max().item()),
        }

    std = samples.std(dim=0, unbiased=False).clamp_min(std_floor)
    threshold = (0.5 * std).clamp(threshold_min, threshold_max)

    return threshold, {
        "samples": int(samples.size(0)),
        "std_min": float(std.min().item()),
        "std_mean": float(std.mean().item()),
        "std_max": float(std.max().item()),
        "threshold_min": float(threshold.min().item()),
        "threshold_mean": float(threshold.mean().item()),
        "threshold_max": float(threshold.max().item()),
    }


@torch.no_grad()
def calibrate_thresholds(raw_model: SparseMamba2DeltaASRForCTC, calib_loader, device, args):
    was_training = raw_model.training
    raw_model.eval()

    input_fallbacks = raw_model.get_layer_input_thresholds()
    output_fallbacks = raw_model.get_layer_output_thresholds()

    input_samples: List[Optional[torch.Tensor]] = [None] * len(input_fallbacks)
    output_samples: List[Optional[torch.Tensor]] = [None] * len(output_fallbacks)

    seen_batches = 0
    for batch_idx, batch in enumerate(calib_loader):
        if batch_idx >= args.calibration_batches:
            break

        inputs = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)

        _ = raw_model(
            input_ids=inputs,
            input_lengths=input_lengths,
            apply_qat_quantization=False,
        )
        output_lengths = _compute_output_lengths(raw_model, input_lengths)

        for layer_idx, layer in enumerate(raw_model.sparse_mamba_delta_asr.layers):
            mixer = layer.mixer
            input_rows = _sample_valid_rows(
                mixer.last_input_prequant,
                output_lengths,
                args.sample_rows_per_batch,
            )
            output_rows = _sample_valid_rows(
                mixer.last_output_prequant,
                output_lengths,
                args.sample_rows_per_batch,
            )
            input_samples[layer_idx] = _merge_row_samples(
                input_samples[layer_idx],
                input_rows,
                args.max_rows_per_layer,
            )
            output_samples[layer_idx] = _merge_row_samples(
                output_samples[layer_idx],
                output_rows,
                args.max_rows_per_layer,
            )

        seen_batches += 1

    if seen_batches == 0:
        raise RuntimeError("Calibration saw zero batches. Increase --calibration_batches.")

    layer_input_thresholds = []
    layer_output_thresholds = []
    report = []

    for layer_idx, (input_rows, output_rows, input_fallback, output_fallback) in enumerate(
        zip(input_samples, output_samples, input_fallbacks, output_fallbacks)
    ):
        input_thresholds, input_stats = _thresholds_from_std_samples(
            input_rows,
            input_fallback,
            args.threshold_min,
            args.threshold_max,
            args.std_floor,
        )
        output_thresholds, output_stats = _thresholds_from_std_samples(
            output_rows,
            output_fallback,
            args.threshold_min,
            args.threshold_max,
            args.std_floor,
        )

        layer_input_thresholds.append(input_thresholds)
        layer_output_thresholds.append(output_thresholds)
        report.append(
            {
                "layer": layer_idx,
                "input": input_stats,
                "output": output_stats,
            }
        )

    raw_model.set_layer_input_thresholds(layer_input_thresholds)
    raw_model.set_layer_output_thresholds(layer_output_thresholds)

    if was_training:
        raw_model.train()

    return report


def broadcast_thresholds(raw_model: SparseMamba2DeltaASRForCTC):
    for layer in raw_model.sparse_mamba_delta_asr.layers:
        dist.broadcast(layer.mixer._input_threshold_raw.data, src=0)
        dist.broadcast(layer.mixer._output_threshold_raw.data, src=0)


def _zero_rate_from_discrete(
    discrete: Optional[torch.Tensor],
    output_lengths: torch.Tensor,
) -> Optional[float]:
    if discrete is None:
        return None
    batch_size, seq_len, hidden_dim = discrete.shape
    positions = torch.arange(seq_len, device=discrete.device).unsqueeze(0)
    valid = positions < output_lengths.to(discrete.device).unsqueeze(1)
    valid = valid.unsqueeze(-1).expand(batch_size, seq_len, hidden_dim)
    discrete_prev = torch.cat([torch.zeros_like(discrete[:, :1, :]), discrete[:, :-1, :]], dim=1)
    rounded_diff = discrete - discrete_prev
    if not torch.any(valid):
        return None
    return float((rounded_diff == 0)[valid].float().mean().item() * 100.0)


@torch.no_grad()
def _collect_zero_rates(
    raw_model: SparseMamba2DeltaASRForCTC,
    output_lengths: torch.Tensor,
) -> Tuple[float, float]:
    input_zero_rates = []
    output_zero_rates = []

    for layer in raw_model.sparse_mamba_delta_asr.layers:
        mixer = layer.mixer
        input_zero_rate = _zero_rate_from_discrete(
            mixer.last_input_discrete,
            output_lengths,
        )
        output_zero_rate = _zero_rate_from_discrete(
            mixer.last_output_discrete,
            output_lengths,
        )
        if input_zero_rate is not None:
            input_zero_rates.append(input_zero_rate)
        if output_zero_rate is not None:
            output_zero_rates.append(output_zero_rate)

    mean_input_zero_rate = sum(input_zero_rates) / len(input_zero_rates) if input_zero_rates else 0.0
    mean_output_zero_rate = sum(output_zero_rates) / len(output_zero_rates) if output_zero_rates else 0.0
    return mean_input_zero_rate, mean_output_zero_rate


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    max_batches: int = 30,
    apply_qat_quantization: bool = True,
):
    raw_model = model.module if isinstance(model, DDP) else model
    was_training = raw_model.training
    raw_model.eval()

    total_ctc = 0.0
    total_reg = 0.0
    input_zero_rates = []
    output_zero_rates = []
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        inputs = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        labels = batch["targets"].to(device, dtype=torch.long)
        input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

        output = raw_model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            apply_qat_quantization=apply_qat_quantization,
        )

        if output.loss is not None:
            total_ctc += float(output.loss.item())
        if output.regularization_term is not None:
            total_reg += float(output.regularization_term.item())

        output_lengths = _compute_output_lengths(raw_model, input_lengths)
        input_zero_rate, output_zero_rate = _collect_zero_rates(raw_model, output_lengths)
        input_zero_rates.append(input_zero_rate)
        output_zero_rates.append(output_zero_rate)

        preds = decode_predictions_ctc(output.logits, output_lengths, blank_id=raw_model.blank_id)
        all_preds.extend(preds)

        start = 0
        for sample_idx in range(len(preds)):
            target_len = target_lengths[sample_idx].item()
            all_targets.append(labels[start:start + target_len].tolist())
            start += target_len

    num_batches = max(1, min(len(dataloader), max_batches if max_batches > 0 else len(dataloader)))
    metrics = calculate_asr_metrics(all_preds, all_targets, dataloader.idx_to_char)

    if was_training:
        raw_model.train()
    else:
        raw_model.eval()

    return {
        "ctc": total_ctc / num_batches,
        "reg": total_reg / num_batches,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "input_zero_rate": sum(input_zero_rates) / len(input_zero_rates) if input_zero_rates else 0.0,
        "output_zero_rate": sum(output_zero_rates) / len(output_zero_rates) if output_zero_rates else 0.0,
    }


def initialize_thresholds(model: SparseMamba2DeltaASRForCTC, value: float = 1.0):
    input_thresholds = [
        torch.full_like(threshold, value)
        for threshold in model.get_layer_input_thresholds()
    ]
    output_thresholds = [
        torch.full_like(threshold, value)
        for threshold in model.get_layer_output_thresholds()
    ]
    model.set_layer_input_thresholds(input_thresholds)
    model.set_layer_output_thresholds(output_thresholds)


def _valid_token_mask(output_lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < output_lengths.to(device).unsqueeze(1)


def _masked_mean_abs(tensor: Optional[torch.Tensor], output_lengths: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    mask = _valid_token_mask(output_lengths, tensor.size(1), tensor.device).to(tensor.dtype)
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * tensor.size(-1)
    return (tensor.abs() * mask).sum() / denom


def _masked_temporal_diff_l1(
    tensor: Optional[torch.Tensor],
    output_lengths: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    prev = torch.cat([torch.zeros_like(tensor[:, :1, :]), tensor[:, :-1, :]], dim=1)
    return _masked_mean_abs(tensor - prev, output_lengths)


def compute_regularization_losses(
    student_model: SparseMamba2DeltaASRForCTC,
    output_lengths: torch.Tensor,
) -> torch.Tensor:
    per_layer_sparse_l1 = []

    for student_layer in student_model.sparse_mamba_delta_asr.layers:
        student_mixer = student_layer.mixer
        per_layer_sparse_terms = []
        input_sparse = _masked_temporal_diff_l1(student_mixer.last_input_discrete, output_lengths)
        output_sparse = _masked_temporal_diff_l1(student_mixer.last_output_discrete, output_lengths)
        if input_sparse is not None:
            per_layer_sparse_terms.append(input_sparse)
        if output_sparse is not None:
            per_layer_sparse_terms.append(output_sparse)
        if per_layer_sparse_terms:
            per_layer_sparse_l1.append(torch.stack(per_layer_sparse_terms).mean())

    device = output_lengths.device
    sparse_l1 = (
        torch.stack(per_layer_sparse_l1).mean()
        if per_layer_sparse_l1
        else torch.zeros((), device=device)
    )
    return sparse_l1


def build_optimizer(raw_model: SparseMamba2DeltaASRForCTC, args):
    threshold_params = [p for p in raw_model.get_threshold_parameters() if p.requires_grad]
    threshold_param_ids = {id(p) for p in threshold_params}
    base_params = [
        p for p in raw_model.parameters()
        if p.requires_grad and id(p) not in threshold_param_ids
    ]

    param_groups = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    if threshold_params:
        param_groups.append(
            {
                "params": threshold_params,
                "lr": args.threshold_lr,
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    return optimizer, scheduler


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, args, extra: dict):
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


def finetune(args):
    local_rank, global_rank, world_size, device = setup_ddp()
    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.enable_wandb and not HAS_WANDB and is_main_process():
            print("WandB logging requested but `wandb` is not installed. Continuing without WandB.")
        maybe_init_wandb(args)

        train_loader, val_loader, test_loader, calib_loader, train_sampler = build_dataloaders(
            args,
            world_size,
            global_rank,
        )

        config = load_config(args.config_path, initial_threshold=args.initial_threshold)
        student = SparseMamba2DeltaASRForCTC(config).to(device)
        student = load_pretrained(student, args.checkpoint)

        baseline_metrics = None
        student_init_metrics = None
        calibration_report = []

        if is_main_process():
            print("Evaluating pretrained baseline with quantization disabled...")
            baseline_metrics = evaluate_model(
                student,
                val_loader,
                device,
                max_batches=args.eval_batches,
                apply_qat_quantization=False,
            )
            print(
                f"  baseline val CTC={baseline_metrics['ctc']:.4f} "
                f"CER={baseline_metrics['cer']:.4f} "
                f"WER={baseline_metrics['wer']:.4f}"
            )
            if wandb_enabled():
                wandb.log(
                    {
                        "baseline/val_ctc": baseline_metrics["ctc"],
                        "baseline/val_reg": baseline_metrics["reg"],
                        "baseline/val_cer": baseline_metrics["cer"],
                        "baseline/val_wer": baseline_metrics["wer"],
                        "baseline/input_zero_rate": baseline_metrics["input_zero_rate"],
                        "baseline/output_zero_rate": baseline_metrics["output_zero_rate"],
                    },
                    step=0,
                )

            if calib_loader is None:
                raise RuntimeError("Calibration loader is not available on rank 0.")
            print(f"Running calibration on {args.calibration_source} with k = 0.5*std(channel)...")
            calibration_report = calibrate_thresholds(student, calib_loader, device, args)
            with open(output_dir / "calibration_report.json", "w") as f:
                json.dump(calibration_report, f, indent=2)
            torch.save(
                {
                    "args": vars(args),
                    "calibration_report": calibration_report,
                    "layer_input_thresholds": [t.cpu() for t in student.get_layer_input_thresholds()],
                    "layer_output_thresholds": [t.cpu() for t in student.get_layer_output_thresholds()],
                },
                output_dir / "calibrated_thresholds.pt",
            )
            for row in calibration_report:
                print(
                    f"  layer {row['layer']:02d} | "
                    f"in_std={row['input']['std_mean']:.5f} "
                    f"out_std={row['output']['std_mean']:.5f} "
                    f"in_k={row['input']['threshold_mean']:.5f} "
                    f"out_k={row['output']['threshold_mean']:.5f}"
                )
            if wandb_enabled():
                _log_calibration_report_wandb(calibration_report, step=0)
                _log_thresholds_wandb(student, step=0, prefix="thresholds/calibrated")

            print("Evaluating calibrated quantized student...")
            student_init_metrics = evaluate_model(
                student,
                val_loader,
                device,
                max_batches=args.eval_batches,
                apply_qat_quantization=True,
            )
            print(
                f"  student val CTC={student_init_metrics['ctc']:.4f} "
                f"CER={student_init_metrics['cer']:.4f} "
                f"WER={student_init_metrics['wer']:.4f} "
                f"in_zero={student_init_metrics['input_zero_rate']:.2f}% "
                f"out_zero={student_init_metrics['output_zero_rate']:.2f}%"
            )
            if wandb_enabled():
                wandb.log(
                    {
                        "student_init/val_ctc": student_init_metrics["ctc"],
                        "student_init/val_reg": student_init_metrics["reg"],
                        "student_init/val_cer": student_init_metrics["cer"],
                        "student_init/val_wer": student_init_metrics["wer"],
                        "student_init/input_zero_rate": student_init_metrics["input_zero_rate"],
                        "student_init/output_zero_rate": student_init_metrics["output_zero_rate"],
                    },
                    step=0,
                )

        broadcast_thresholds(student)
        dist.barrier()

        for param in student.parameters():
            param.requires_grad_(True)
        student.set_threshold_trainable(True)

        raw_student = student
        optimizer, scheduler = build_optimizer(raw_student, args)
        amp_dtype = None if args.mixed_precision == "none" else getattr(torch, args.mixed_precision)
        scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "float16")

        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        raw_student = student.module

        best_cer = float("inf")
        train_sampler.set_epoch(0)
        train_iter = iter(train_loader)

        if is_main_process():
            print(f"Starting task-loss finetuning for {args.steps} steps...")

        for step in range(1, args.steps + 1):
            student.train()
            raw_student.sparse_mamba_delta_asr.spec_augment.eval()

            try:
                batch = next(train_iter)
            except StopIteration:
                train_sampler.set_epoch(step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            inputs = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
            labels = batch["targets"].to(device, dtype=torch.long)
            input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)
            target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            l1_schedule_steps = (
                args.sparsity_l1_schedule_steps
                if args.sparsity_l1_schedule_steps is not None
                else args.warmup_steps
            )
            current_sparsity_l1_weight = schedule_value(
                max(step - 1, 0),
                l1_schedule_steps,
                args.sparsity_l1_weight_start,
                args.sparsity_l1_weight,
            )

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                student_output = student(
                    input_ids=inputs,
                    labels=labels,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                    apply_qat_quantization=True,
                )
                output_lengths = student_output.output_lengths
                if output_lengths is None:
                    output_lengths = _compute_output_lengths(raw_student, input_lengths)
                task_loss = student_output.loss
                if task_loss is None:
                    raise RuntimeError("Student output did not provide the usual task loss.")
                sparsity_l1 = compute_regularization_losses(
                    raw_student,
                    output_lengths,
                )
                base_loss = args.ctc_weight * task_loss
                loss = base_loss + current_sparsity_l1_weight * sparsity_l1

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

            stats = torch.tensor(
                [task_loss.detach(), sparsity_l1.detach(), loss.detach()],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            mean_task, mean_sparsity_l1, mean_loss = [x.item() for x in stats]

            if step % args.log_freq == 0 and is_main_process():
                print(
                    f"step={step:5d}/{args.steps} "
                    f"task={mean_task:.4f} sparsity_l1={mean_sparsity_l1:.4f} "
                    f"loss={mean_loss:.4f} "
                    f"l1_weight={current_sparsity_l1_weight:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"grad_norm={float(grad_norm):.3f}"
                )
                if wandb_enabled():
                    wandb.log(
                        {
                            "train/task": mean_task,
                            "train/sparsity_l1": mean_sparsity_l1,
                            "train/loss": mean_loss,
                            "train/sparsity_l1_weight": current_sparsity_l1_weight,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": float(grad_norm),
                        },
                        step=step,
                    )

            dist.barrier()

            if step % args.val_freq == 0 and is_main_process():
                val_metrics = evaluate_model(
                    student,
                    val_loader,
                    device,
                    max_batches=args.eval_batches,
                    apply_qat_quantization=True,
                )
                print(
                    f"  [val] step={step} "
                    f"CTC={val_metrics['ctc']:.4f} "
                    f"REG={val_metrics['reg']:.4f} "
                    f"CER={val_metrics['cer']:.4f} "
                    f"WER={val_metrics['wer']:.4f} "
                    f"in_zero={val_metrics['input_zero_rate']:.2f}% "
                    f"out_zero={val_metrics['output_zero_rate']:.2f}%"
                )
                if wandb_enabled():
                    wandb.log(
                        {
                            "val/ctc": val_metrics["ctc"],
                            "val/reg": val_metrics["reg"],
                            "val/cer": val_metrics["cer"],
                            "val/wer": val_metrics["wer"],
                            "val/input_zero_rate": val_metrics["input_zero_rate"],
                            "val/output_zero_rate": val_metrics["output_zero_rate"],
                        },
                        step=step,
                    )
                    _log_thresholds_wandb(raw_student, step=step, prefix="thresholds")

                save_checkpoint(
                    output_dir / "latest.pt",
                    student,
                    optimizer,
                    scheduler,
                    step,
                    args,
                    {
                        "val_metrics": val_metrics,
                        "baseline_metrics": baseline_metrics,
                        "student_init_metrics": student_init_metrics,
                        "calibration_report": calibration_report,
                    },
                )

                if val_metrics["cer"] < best_cer:
                    best_cer = val_metrics["cer"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        student,
                        optimizer,
                        scheduler,
                        step,
                        args,
                        {
                            "val_metrics": val_metrics,
                            "baseline_metrics": baseline_metrics,
                            "student_init_metrics": student_init_metrics,
                            "calibration_report": calibration_report,
                        },
                    )
                    print(f"  New best checkpoint saved (CER={best_cer:.4f})")

            dist.barrier()

        dist.barrier()
        if is_main_process():
            print("Running final validation and test evaluation...")
            final_val_metrics = evaluate_model(
                student,
                val_loader,
                device,
                max_batches=max(args.eval_batches, 50),
                apply_qat_quantization=True,
            )
            final_test_metrics = evaluate_model(
                student,
                test_loader,
                device,
                max_batches=max(args.eval_batches, 50),
                apply_qat_quantization=True,
            )

            print(
                f"  final val  CTC={final_val_metrics['ctc']:.4f} "
                f"CER={final_val_metrics['cer']:.4f} "
                f"WER={final_val_metrics['wer']:.4f} "
                f"in_zero={final_val_metrics['input_zero_rate']:.2f}% "
                f"out_zero={final_val_metrics['output_zero_rate']:.2f}%"
            )
            print(
                f"  final test CTC={final_test_metrics['ctc']:.4f} "
                f"CER={final_test_metrics['cer']:.4f} "
                f"WER={final_test_metrics['wer']:.4f} "
                f"in_zero={final_test_metrics['input_zero_rate']:.2f}% "
                f"out_zero={final_test_metrics['output_zero_rate']:.2f}%"
            )

            raw_student = student.module if isinstance(student, DDP) else student
            torch.save(raw_student.state_dict(), output_dir / "final_model.pt")
            torch.save(
                {
                    "args": vars(args),
                    "layer_input_thresholds": [t.cpu() for t in raw_student.get_layer_input_thresholds()],
                    "layer_output_thresholds": [t.cpu() for t in raw_student.get_layer_output_thresholds()],
                    "baseline_metrics": baseline_metrics,
                    "student_init_metrics": student_init_metrics,
                    "calibration_report": calibration_report,
                    "final_val_metrics": final_val_metrics,
                    "final_test_metrics": final_test_metrics,
                },
                output_dir / "final_thresholds.pt",
            )
            if wandb_enabled():
                wandb.log(
                    {
                        "final/val_ctc": final_val_metrics["ctc"],
                        "final/val_reg": final_val_metrics["reg"],
                        "final/val_cer": final_val_metrics["cer"],
                        "final/val_wer": final_val_metrics["wer"],
                        "final/val_input_zero_rate": final_val_metrics["input_zero_rate"],
                        "final/val_output_zero_rate": final_val_metrics["output_zero_rate"],
                        "final/test_ctc": final_test_metrics["ctc"],
                        "final/test_reg": final_test_metrics["reg"],
                        "final/test_cer": final_test_metrics["cer"],
                        "final/test_wer": final_test_metrics["wer"],
                        "final/test_input_zero_rate": final_test_metrics["input_zero_rate"],
                        "final/test_output_zero_rate": final_test_metrics["output_zero_rate"],
                    },
                    step=args.steps,
                )
                _log_thresholds_wandb(raw_student, step=args.steps, prefix="thresholds/final")
        dist.barrier()
    finally:
        if wandb_enabled() and is_main_process():
            wandb.finish()
        cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", default="./qat_delta_ckpts")
    parser.add_argument("--cache_dir", required=True)

    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--initial_threshold", type=float, default=None)
    parser.add_argument(
        "--calibration_source",
        choices=["validation", "train_subset"],
        default="train_subset",
    )
    parser.add_argument("--calibration_batches", type=int, default=64)
    parser.add_argument("--calibration_train_samples", type=int, default=16000)
    parser.add_argument("--calibration_seed", type=int, default=1234)
    parser.add_argument("--sample_rows_per_batch", type=int, default=128)
    parser.add_argument("--max_rows_per_layer", type=int, default=4096)
    parser.add_argument("--threshold_min", type=float, default=1e-4)
    parser.add_argument("--threshold_max", type=float, default=1e4)
    parser.add_argument("--std_floor", type=float, default=1e-4)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threshold_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ctc_weight", type=float, default=1.0)
    parser.add_argument("--sparsity_l1_weight", type=float, default=0.05)
    parser.add_argument("--sparsity_l1_weight_start", type=float, default=0.0)
    parser.add_argument("--sparsity_l1_schedule_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed_precision",
        choices=["none", "float16", "bfloat16"],
        default="bfloat16",
    )

    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=30)
    parser.add_argument("--enable_wandb", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
