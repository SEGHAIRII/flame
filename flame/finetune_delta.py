#!/usr/bin/env python3
"""
Calibration + threshold finetuning for SparseMamba2DeltaASRForCTC (DDP).

Workflow:
1) Run calibration with apply_thresholding=False to collect score statistics
   s_i = |diff_i| * ||W[:, i]||_2
2) Choose a score threshold t from calibration percentiles.
3) Convert to per-channel thresholds tau_i = t / (||W[:, i]||_2 + eps).
4) Save thresholds and finetune with apply_thresholding=True using task loss only.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
    SparseMamba2DeltaASRConfig,
    SparseMamba2DeltaASRForCTC,
)
from sparse_mamba.custom_dataloaders.preprocess_ctc import CachedASRDataset, collate_fn_ctc
from flame.train_parallel_asr import calculate_asr_metrics, decode_predictions_ctc

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, global_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str, initial_threshold: float) -> SparseMamba2DeltaASRConfig:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["threshold"] = float(initial_threshold)
    cfg["return_activation_sparsity"] = True
    return SparseMamba2DeltaASRConfig(**cfg)


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
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds if test_ds is not None else train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optional calibration loader from a deterministic subset of training data (rank-0 only).
    calib_loader = None
    if global_rank == 0 and args.calibration_source == "train_subset":
        n_train = len(train_ds)
        n_calib = args.calibration_train_samples
        if n_calib <= 0:
            n_calib = min(n_train, args.calibration_batches * args.batch_size)
        n_calib = min(n_train, n_calib)

        g = torch.Generator()
        g.manual_seed(args.calibration_seed)
        indices = torch.randperm(n_train, generator=g)[:n_calib].tolist()
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


def _extract_state_dict(ckpt_obj):
    if "model_state_dicts" in ckpt_obj:
        return ckpt_obj["model_state_dicts"][0]
    if "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_pretrained(model: SparseMamba2DeltaASRForCTC, checkpoint_path: str, drop_channel_thresholds: bool = False):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)

    dropped = [k for k in list(sd.keys()) if "_threshold_raw" in k]
    if drop_channel_thresholds:
        dropped.extend([k for k in list(sd.keys()) if "channel_thresholds" in k])
    for k in dropped:
        sd.pop(k, None)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if dropped:
        print(f"  Dropped {len(dropped)} threshold keys from checkpoint")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model


# ---------------------------------------------------------------------------
# Metrics / evaluation
# ---------------------------------------------------------------------------


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


@torch.no_grad()
def _masked_layer_sparsities(
    raw_model: SparseMamba2DeltaASRForCTC, output_lengths: torch.Tensor, apply_thresholding: bool
) -> List[float]:
    layer_sparsities = []
    encoder = getattr(raw_model, "sparse_mamba_delta_asr", None)
    if encoder is None:
        return layer_sparsities

    for layer in encoder.layers:
        diff = getattr(layer.mixer, "last_pre_threshold", None)
        if diff is None:
            continue

        delta = layer.mixer._apply_channel_threshold(diff, apply_thresholding=apply_thresholding)
        batch_size, seq_len, hidden_dim = delta.shape

        valid_bt = (
            torch.arange(seq_len, device=delta.device).view(1, seq_len, 1)
            < output_lengths.to(delta.device).view(batch_size, 1, 1)
        )
        valid_mask = valid_bt.expand(-1, -1, hidden_dim)
        if not torch.any(valid_mask):
            continue

        layer_sp = (delta[valid_mask] == 0).float().mean() * 100.0
        layer_sparsities.append(float(layer_sp.item()))

    return layer_sparsities


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_batches=30, apply_thresholding=True):
    raw_model = model.module if isinstance(model, DDP) else model
    was_training = raw_model.training
    raw_model.eval()

    total_ctc = 0.0
    all_sparsities = []
    all_preds = []
    all_targets = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        inputs = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        labels = batch["targets"].to(device, dtype=torch.long)
        input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

        out = raw_model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            apply_thresholding=apply_thresholding,
        )

        if out.loss is not None:
            total_ctc += out.loss.item()

        output_lengths = _compute_output_lengths(raw_model, input_lengths)
        masked_sps = _masked_layer_sparsities(raw_model, output_lengths, apply_thresholding=apply_thresholding)
        if masked_sps:
            all_sparsities.extend(masked_sps)
        elif out.all_sparsities:
            for s in out.all_sparsities:
                all_sparsities.append(float(s.item()) if torch.is_tensor(s) else float(s))

        preds = decode_predictions_ctc(out.logits, output_lengths)
        all_preds.extend(preds)

        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t

    n = max(1, min(max_batches, len(dataloader)))
    avg_ctc = total_ctc / n
    avg_sparsity = sum(all_sparsities) / len(all_sparsities) if all_sparsities else 0.0
    metrics = calculate_asr_metrics(all_preds, all_targets, dataloader.idx_to_char)

    if was_training:
        raw_model.train()
    else:
        raw_model.eval()

    return avg_ctc, avg_sparsity, metrics["cer"], metrics["wer"]


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------


def _log_thresholds_wandb(raw_model: SparseMamba2DeltaASRForCTC, step: int, prefix: str = "thresholds"):
    if not HAS_WANDB:
        return
    layer_thresholds = raw_model.get_layer_channel_thresholds()
    if not layer_thresholds:
        return

    log_data = {}
    flat = torch.cat([t.detach().cpu().float() for t in layer_thresholds], dim=0)
    log_data[f"{prefix}/global_mean"] = float(flat.mean().item())
    log_data[f"{prefix}/global_min"] = float(flat.min().item())
    log_data[f"{prefix}/global_max"] = float(flat.max().item())
    log_data[f"{prefix}/global_hist"] = wandb.Histogram(flat.numpy())

    for i, t in enumerate(layer_thresholds):
        tt = t.detach().cpu().float()
        layer_name = f"{prefix}/layer_{i:02d}"
        log_data[f"{layer_name}/mean"] = float(tt.mean().item())
        log_data[f"{layer_name}/min"] = float(tt.min().item())
        log_data[f"{layer_name}/max"] = float(tt.max().item())
        log_data[f"{layer_name}/hist"] = wandb.Histogram(tt.numpy())

    wandb.log(log_data, step=step)


def _log_calibration_report_wandb(calibration_report, step: int):
    if not HAS_WANDB:
        return
    layer_rows = [r for r in calibration_report if isinstance(r, dict) and "layer" in r]
    if not layer_rows:
        return

    score_thresholds = [float(r["score_threshold"]) for r in layer_rows]
    predicted_sps = [float(r["predicted_sparsity"]) for r in layer_rows]
    tau_means = [float(r["tau_mean"]) for r in layer_rows]

    wandb.log(
        {
            "calibration/score_threshold_mean": sum(score_thresholds) / len(score_thresholds),
            "calibration/score_threshold_min": min(score_thresholds),
            "calibration/score_threshold_max": max(score_thresholds),
            "calibration/predicted_sparsity_mean": sum(predicted_sps) / len(predicted_sps),
            "calibration/predicted_sparsity_min": min(predicted_sps),
            "calibration/predicted_sparsity_max": max(predicted_sps),
            "calibration/tau_mean_mean": sum(tau_means) / len(tau_means),
            "calibration/tau_mean_min": min(tau_means),
            "calibration/tau_mean_max": max(tau_means),
        },
        step=step,
    )

    table = wandb.Table(
        columns=["layer", "scores", "score_threshold", "predicted_sparsity", "tau_min", "tau_mean", "tau_max"]
    )
    for r in layer_rows:
        table.add_data(
            int(r["layer"]),
            int(r["scores"]),
            float(r["score_threshold"]),
            float(r["predicted_sparsity"]),
            float(r["tau_min"]),
            float(r["tau_mean"]),
            float(r["tau_max"]),
        )
    wandb.log({"calibration/report_table": table}, step=step)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _sample_scores(x: torch.Tensor, sample_count: int) -> torch.Tensor:
    if x.numel() <= sample_count:
        return x
    idx = torch.randperm(x.numel(), device=x.device)[:sample_count]
    return x[idx]


@torch.no_grad()
def calibrate_thresholds(raw_model: SparseMamba2DeltaASRForCTC, calib_loader, device, args):
    was_training = raw_model.training
    raw_model.eval()

    layer_col_norms = raw_model.get_layer_outproj_column_norms(eps=args.threshold_eps)
    n_layers = len(layer_col_norms)
    score_samples: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]

    seen = 0
    for i, batch in enumerate(calib_loader):
        if i >= args.calibration_batches:
            break

        inputs = batch["features"].to(device, dtype=torch.float32).transpose(1, 2)
        labels = batch["targets"].to(device, dtype=torch.long)
        input_lengths = batch["feature_lengths"].to(device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(device, dtype=torch.long)

        _ = raw_model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            apply_thresholding=False,
        )
        output_lengths = _compute_output_lengths(raw_model, input_lengths)

        for layer_idx, layer in enumerate(raw_model.sparse_mamba_delta_asr.layers):
            diff = layer.mixer.last_pre_threshold
            if diff is None:
                continue

            c = layer_col_norms[layer_idx].to(device=diff.device, dtype=diff.dtype).view(1, 1, -1)
            scores = diff.abs() * c
            batch_size, seq_len, hidden_dim = scores.shape
            valid_bt = (
                torch.arange(seq_len, device=scores.device).view(1, seq_len, 1)
                < output_lengths.to(scores.device).view(batch_size, 1, 1)
            )
            scores = scores[valid_bt.expand(-1, -1, hidden_dim)]
            if scores.numel() == 0:
                continue
            scores = _sample_scores(scores, args.score_samples_per_batch)
            score_samples[layer_idx].append(scores.detach().cpu())

        seen += 1

    if seen == 0:
        raise RuntimeError("Calibration saw zero batches. Increase --calibration_batches.")

    q = max(0.0, min(1.0, args.target_sparsity / 100.0))
    layer_thresholds = []
    report = []

    for layer_idx in range(n_layers):
        c_cpu = layer_col_norms[layer_idx].detach().cpu()

        if not score_samples[layer_idx]:
            score_t = 0.0
            tau = torch.zeros_like(c_cpu)
            predicted_sp = 0.0
            num_scores = 0
        else:
            scores = torch.cat(score_samples[layer_idx], dim=0)
            if scores.numel() > args.max_score_samples_per_layer:
                keep = _sample_scores(scores, args.max_score_samples_per_layer)
                scores = keep.cpu()

            if args.manual_score_threshold is not None:
                score_t = float(args.manual_score_threshold)
            else:
                score_t = float(torch.quantile(scores, q).item())

            predicted_sp = float((scores <= score_t).float().mean().item() * 100.0)
            tau = (score_t / (c_cpu + args.threshold_eps)).clamp_min(0.0)
            num_scores = int(scores.numel())

        layer_thresholds.append(tau)
        report.append(
            {
                "layer": layer_idx,
                "scores": num_scores,
                "score_threshold": score_t,
                "predicted_sparsity": predicted_sp,
                "tau_min": float(tau.min().item()),
                "tau_mean": float(tau.mean().item()),
                "tau_max": float(tau.max().item()),
            }
        )

    raw_model.set_layer_channel_thresholds(layer_thresholds)

    if was_training:
        raw_model.train()

    return report


@torch.no_grad()
def load_thresholds_from_file(raw_model: SparseMamba2DeltaASRForCTC, path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "layer_channel_thresholds" in payload:
        layer_thresholds = payload["layer_channel_thresholds"]
    elif isinstance(payload, list):
        layer_thresholds = payload
    elif isinstance(payload, dict):
        layer_keys = sorted(
            [k for k in payload.keys() if k.endswith("channel_thresholds")],
            key=lambda x: int(x.split("layers.")[1].split(".")[0]),
        )
        layer_thresholds = [payload[k] for k in layer_keys]
    else:
        raise ValueError(f"Unsupported thresholds format in {path}")

    tensors = [torch.as_tensor(t, dtype=torch.float32) for t in layer_thresholds]
    raw_model.set_layer_channel_thresholds(tensors)


def broadcast_thresholds(raw_model: SparseMamba2DeltaASRForCTC):
    for layer in raw_model.sparse_mamba_delta_asr.layers:
        dist.broadcast(layer.mixer.channel_thresholds, src=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def finetune(args):
    local_rank, global_rank, world_size, device = setup_ddp()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(f"Device: {device} | World size: {world_size}")
        print("Loading dataset...")

    train_loader, val_loader, test_loader, calib_loader, train_sampler = build_dataloaders(
        args, world_size, global_rank
    )

    if is_main_process():
        print("Building model...")
    config = load_config(args.config_path, args.initial_threshold)
    model = SparseMamba2DeltaASRForCTC(config).to(device)
    model.eval()

    if is_main_process():
        print("Loading checkpoint...")
    model = load_pretrained(model, args.checkpoint, drop_channel_thresholds=args.overwrite_saved_thresholds)

    if HAS_WANDB and is_main_process() and args.enable_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "asr"),
            name=os.environ.get("WANDB_NAME", "finetune_delta"),
            config=vars(args),
        )

    # Baseline with no thresholding
    if is_main_process():
        print("Evaluating baseline on test split (apply_thresholding=False)...")
        model.eval()
        base_ctc, base_sp, base_cer, base_wer = evaluate_model(
            model,
            test_loader,
            device,
            max_batches=args.eval_batches,
            apply_thresholding=False,
        )
        print(
            f"  Baseline CTC={base_ctc:.4f} CER={base_cer:.4f} "
            f"WER={base_wer:.4f} sparsity={base_sp:.2f}%"
        )
        if HAS_WANDB and args.enable_wandb:
            wandb.log(
                {
                    "baseline/no_threshold/ctc": base_ctc,
                    "baseline/no_threshold/cer": base_cer,
                    "baseline/no_threshold/wer": base_wer,
                    "baseline/no_threshold/sparsity": base_sp,
                },
                step=0,
            )
    else:
        base_ctc, base_sp, base_cer, base_wer = 0.0, 0.0, 0.0, 0.0

    baseline_tensor = torch.tensor([base_ctc, base_sp, base_cer, base_wer], device=device)
    dist.broadcast(baseline_tensor, src=0)
    base_ctc, base_sp, base_cer, base_wer = [x.item() for x in baseline_tensor]

    # Threshold assignment (manual file or calibration)
    raw_model = model
    calibration_report = []
    calibration_source_name = "unknown"

    if is_main_process():
        if args.thresholds_path:
            print(f"Loading manual thresholds from {args.thresholds_path}...")
            load_thresholds_from_file(raw_model, args.thresholds_path)
            calibration_source_name = "file"
            calibration_report = [{"source": "file", "path": args.thresholds_path}]
        else:
            raw_model.eval()
            calibration_loader = val_loader
            calibration_source_name = "validation"
            if args.calibration_source == "train_subset":
                if calib_loader is None:
                    raise RuntimeError("Calibration source is train_subset but calibration loader is not available.")
                calibration_loader = calib_loader
                calibration_source_name = f"train_subset(n={len(calibration_loader.dataset)})"

            print(f"Running calibration on {calibration_source_name} (apply_thresholding=False)...")
            calibration_report = calibrate_thresholds(raw_model, calibration_loader, device, args)
            for r in calibration_report:
                print(
                    f"  Layer {r['layer']:02d} | score_t={r['score_threshold']:.6f} "
                    f"pred_sp={r['predicted_sparsity']:.2f}% "
                    f"tau_mean={r['tau_mean']:.6f}"
                )

            calib_payload = {
                "args": vars(args),
                "calibration_report": calibration_report,
                "calibration_source": calibration_source_name,
                "layer_channel_thresholds": [t.cpu() for t in raw_model.get_layer_channel_thresholds()],
            }
            torch.save(calib_payload, output_dir / "calibrated_thresholds.pt")
            with open(output_dir / "calibration_report.json", "w") as f:
                json.dump(calibration_report, f, indent=2)
            print(f"Saved calibration thresholds to {output_dir / 'calibrated_thresholds.pt'}")

        if HAS_WANDB and args.enable_wandb:
            wandb.log({"calibration/source": calibration_source_name}, step=0)
            _log_calibration_report_wandb(calibration_report, step=0)
            _log_thresholds_wandb(raw_model, step=0, prefix="thresholds/calibrated")

    # Synchronize thresholds to all ranks
    broadcast_thresholds(raw_model)
    dist.barrier()

    if is_main_process():
        print("Evaluating calibrated model (apply_thresholding=True)...")
        cal_ctc, cal_sp, cal_cer, cal_wer = evaluate_model(
            raw_model,
            val_loader,
            device,
            max_batches=args.eval_batches,
            apply_thresholding=True,
        )
        print(
            f"  Calibrated CTC={cal_ctc:.4f} CER={cal_cer:.4f} "
            f"WER={cal_wer:.4f} sparsity={cal_sp:.2f}%"
        )
        if HAS_WANDB and args.enable_wandb:
            wandb.log(
                {
                    "calibrated/thresholded/ctc": cal_ctc,
                    "calibrated/thresholded/cer": cal_cer,
                    "calibrated/thresholded/wer": cal_wer,
                    "calibrated/thresholded/sparsity": cal_sp,
                },
                step=0,
            )
    dist.barrier()

    # DDP wrap after thresholds are finalized
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    if is_main_process():
        print(f"Starting finetuning for {args.steps} steps (CTC-only)...")

    best_cer = float("inf")
    train_sampler.set_epoch(0)
    train_iter = iter(train_loader)

    for step in range(1, args.steps + 1):
        model.train()

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

        output = model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            apply_thresholding=True,
        )

        ctc_loss = output.loss
        loss = args.ctc_weight * ctc_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        train_stats = torch.tensor(
            [ctc_loss.detach(), loss.detach()],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(train_stats, op=dist.ReduceOp.AVG)
        ctc_val, loss_val = [x.item() for x in train_stats]

        if step % args.log_freq == 0 and is_main_process():
            print(
                f"step={step:5d}/{args.steps} "
                f"ctc={ctc_val:.4f} loss={loss_val:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={float(grad_norm):.3f}"
            )
            if HAS_WANDB and args.enable_wandb:
                wandb.log(
                    {
                        "train/ctc": ctc_val,
                        "train/loss": loss_val,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm),
                    },
                    step=step,
                )

        dist.barrier()

        if step % args.val_freq == 0 and is_main_process():
            val_ctc, val_sp, val_cer, val_wer = evaluate_model(
                model,
                val_loader,
                device,
                max_batches=args.eval_batches,
                apply_thresholding=True,
            )
            print(
                f"  [val] step={step} CTC={val_ctc:.4f} CER={val_cer:.4f} "
                f"WER={val_wer:.4f} sparsity={val_sp:.2f}%"
            )

            raw_sd = (model.module if isinstance(model, DDP) else model).state_dict()
            ckpt = {
                "step": step,
                "model_state_dict": raw_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ctc": val_ctc,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "val_sparsity": val_sp,
                "baseline": {
                    "ctc": base_ctc,
                    "cer": base_cer,
                    "wer": base_wer,
                    "sparsity": base_sp,
                },
                "calibration_report": calibration_report,
                "args": vars(args),
            }
            torch.save(ckpt, output_dir / "latest.pt")

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(ckpt, output_dir / "best.pt")
                print(f"  New best checkpoint saved (CER={best_cer:.4f})")

            if HAS_WANDB and args.enable_wandb:
                wandb.log(
                    {
                        "val/ctc": val_ctc,
                        "val/cer": val_cer,
                        "val/wer": val_wer,
                        "val/sparsity": val_sp,
                    },
                    step=step,
                )

        dist.barrier()

    if is_main_process():
        print("Finetuning complete. Running final evaluation...")
        final_ctc, final_sp, final_cer, final_wer = evaluate_model(
            model,
            val_loader,
            device,
            max_batches=max(args.eval_batches, 50),
            apply_thresholding=True,
        )
        print(
            f"  Baseline (no-th) CTC={base_ctc:.4f} CER={base_cer:.4f} WER={base_wer:.4f} sp={base_sp:.2f}%"
        )
        print(
            f"  Final (th)      CTC={final_ctc:.4f} CER={final_cer:.4f} WER={final_wer:.4f} sp={final_sp:.2f}%"
        )

        raw_model = model.module if isinstance(model, DDP) else model
        torch.save(raw_model.state_dict(), output_dir / "final_model.pt")

        payload = {
            "layer_channel_thresholds": [t.cpu() for t in raw_model.get_layer_channel_thresholds()],
            "calibration_report": calibration_report,
            "args": vars(args),
        }
        torch.save(payload, output_dir / "final_thresholds.pt")
        print(f"Saved model and thresholds to {output_dir}")

        if HAS_WANDB and args.enable_wandb:
            _log_thresholds_wandb(raw_model, step=args.steps, prefix="thresholds/final")
            wandb.log(
                {
                    "final/ctc": final_ctc,
                    "final/cer": final_cer,
                    "final/wer": final_wer,
                    "final/sparsity": final_sp,
                }
            )
            wandb.finish()

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--output_dir", default="./threshold_ckpts")
    p.add_argument("--cache_dir", default="./cache")

    p.add_argument("--dataset", default="librispeech")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--initial_threshold", type=float, default=0.0)
    p.add_argument("--overwrite_saved_thresholds", type=int, default=1,
                   help="1 drops threshold buffers from checkpoint before calibration")

    # Calibration
    p.add_argument("--thresholds_path", default=None,
                   help="Optional file with manual layer thresholds. If set, skip calibration.")
    p.add_argument("--calibration_source", type=str, default="train_subset",
                   choices=["validation", "train_subset"],
                   help="Dataset source used to collect calibration score statistics.")
    p.add_argument("--calibration_train_samples", type=int, default=20000,
                   help="Number of training samples used when calibration_source=train_subset. <=0 uses calibration_batches*batch_size.")
    p.add_argument("--calibration_seed", type=int, default=1234,
                   help="Seed for sampling the train subset used in calibration.")
    p.add_argument("--calibration_batches", type=int, default=80)
    p.add_argument("--target_sparsity", type=float, default=90.0,
                   help="Percent of coordinates to prune by score during calibration")
    p.add_argument("--manual_score_threshold", type=float, default=None,
                   help="If set, uses this score threshold t instead of quantile")
    p.add_argument("--score_samples_per_batch", type=int, default=50000)
    p.add_argument("--max_score_samples_per_layer", type=int, default=300000)
    p.add_argument("--threshold_eps", type=float, default=1e-8)

    # Finetuning
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ctc_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging/eval
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--val_freq", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=30)
    p.add_argument("--enable_wandb", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
