#!/usr/bin/env python3
"""
Threshold finetuning for SparseMamba2DeltaASRForCTC.

Goal: learn a per-layer threshold that maximises activation sparsity
      while keeping CTC loss close to the pretrained baseline.

Usage:
    python finetune_thresholds.py \
        --checkpoint path/to/checkpoint.pt \
        --config_path path/to/config.json \
        --dataset librispeech \
        --steps 2000 \
        --sparsity_weight 0.005 \
        --output_dir ./threshold_ckpts
"""

import os
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
    SparseMamba2DeltaASRForCTC,
    SparseMamba2DeltaASRConfig,
    SparseMamba2Delta,
)
from sparse_mamba.custom_dataloaders.librosa import create_librosa_raw_classification_dataset
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path, initial_threshold):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["threshold"] = initial_threshold  # override so _threshold_raw inits correctly
    cfg["return_activation_sparsity"] = True  # must be on for finetuning
    return SparseMamba2DeltaASRConfig(**cfg)



def load_pretrained(model, checkpoint_path):
    """
    Load pretrained weights, tolerating missing _threshold_raw keys
    (they are new and will keep their init values).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Support both raw state_dict and wrapped checkpoint formats
    if "model_state_dicts" in ckpt:
        sd = ckpt["model_state_dicts"][0]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)

    threshold_missing = [k for k in missing if "_threshold_raw" in k]
    real_missing      = [k for k in missing if "_threshold_raw" not in k]

    print(f"  New threshold params (expected): {len(threshold_missing)}")
    if real_missing:
        print(f"  WARNING — other missing keys: {real_missing}")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected}")

    return model


def unlock_thresholds(model):
    """Freeze everything except _threshold_raw parameters."""
    for name, p in model.named_parameters():
        p.requires_grad = "_threshold_raw" in name

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"Unlocked {len(trainable)} threshold parameters:")
    for name, p in trainable:
        print(f"  {name}  (init threshold={F.softplus(p).item():.6f})")

    return model


def get_threshold_params(model):
    return [(n, p) for n, p in model.named_parameters() if "_threshold_raw" in n]


def log_thresholds(model, step):
    """Print per-layer threshold values and sparsity."""
    vals = []
    for name, p in model.named_parameters():
        if "_threshold_raw" in name:
            vals.append(F.softplus(p).item())
    if vals:
        print(f"  Step {step:5d} | thresholds — "
              f"min={min(vals):.4f}  mean={sum(vals)/len(vals):.4f}  max={max(vals):.4f}")
    return vals


def sparsity_incentive(model, weight):
    loss = torch.tensor(0.0)
    for name, p in model.named_parameters():  # name, not p
        if "_threshold_raw" in name and p.requires_grad:
            loss = loss + F.softplus(p)
    return -weight * loss


# ---------------------------------------------------------------------------
# Evaluate baseline before finetuning
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(model, val_loader, device, max_batches=30):
    model.eval()
    total_ctc = 0.0
    total_sparsity = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch['targets'].to(device, dtype=torch.long)
        input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        out = model(input_ids=inputs, labels=labels,
                    input_lengths=input_lengths, target_lengths=target_lengths)
        total_ctc += out.loss.item()
        if out.all_sparsities:
            total_sparsity.extend(out.all_sparsities)

    n = min(max_batches, len(val_loader))
    avg_ctc      = total_ctc / n
    avg_sparsity = sum(total_sparsity) / len(total_sparsity) if total_sparsity else 0.0

    model.eval()
    return avg_ctc, avg_sparsity


# ---------------------------------------------------------------------------
# Main finetuning loop
# ---------------------------------------------------------------------------

def finetune_thresholds(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading dataset...")
    train_loader, val_loader, _, n_classes, _, input_dim, char_to_idx, idx_to_char = \
        create_librosa_raw_classification_dataset(
            bsz=args.batch_size,
            max_samples=args.max_samples,
            num_mfcc=args.num_mfcc,
            cache_dir=args.cache_dir,
            dataset=args.dataset,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
    print(f"Classes: {n_classes}  |  Input dim: {input_dim}")

    # --- Model ---
    print("Building model...")
    config = load_config(args.config_path, args.initial_threshold)
    model  = SparseMamba2DeltaASRForCTC(config).to(device)

    print("Loading pretrained weights...")
    model = load_pretrained(model, args.checkpoint)

    # --- Sanity check: thresholds should equal args.initial_threshold ---
    print("\nSanity check — initial threshold values:")
    for name, p in model.named_parameters():
        if "_threshold_raw" in name:
            v = F.softplus(p).item()
            match = abs(v - args.initial_threshold) < 1e-3
            print(f"  {name}: {v:.6f}  {'OK' if match else 'MISMATCH'}")

    # --- Baseline ---
    print("\nEvaluating pretrained baseline...")
    base_ctc, base_sparsity = evaluate_baseline(model, val_loader, device)
    print(f"  Baseline CTC loss : {base_ctc:.4f}")
    print(f"  Baseline sparsity : {base_sparsity:.2f}%")

    # --- Unlock thresholds ---
    print("\nUnlocking threshold parameters...")
    model = unlock_thresholds(model)

    # --- Optimizer ---
    # Use higher LR than normal — thresholds are scalar params with a
    # relatively smooth loss surface.
    threshold_params = [p for _, p in get_threshold_params(model)]
    optimizer = torch.optim.Adam(threshold_params, lr=args.lr)

    # Cosine anneal LR to zero over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # --- Output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\nStarting threshold finetuning for {args.steps} steps...")
    print(f"  Sparsity weight : {args.sparsity_weight}")
    print(f"  Beta (sigmoid)  : {args.beta} (annealed to {args.beta_max})\n")

    train_iter = iter(train_loader)
    best_score = float('inf')   # lower CTC is better

    # We also track a running CTC to detect if thresholds are
    # hurting performance too much
    running_ctc = base_ctc

    for step in range(1, args.steps + 1):
        model.eval()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch['targets'].to(device, dtype=torch.long)
        input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        # Anneal beta: start soft (easy gradients), end sharp (real threshold)
        beta = args.beta + (args.beta_max - args.beta) * (step / args.steps)

        # Inject current beta into all mixer modules
        for m in model.modules():
            if isinstance(m, SparseMamba2Delta):
                m.threshold_beta = beta

        output = model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        ctc_loss = output.loss

        # Sparsity incentive: reward larger thresholds
        sp_loss = torch.tensor(0.0, device=device)
        for p in threshold_params:
            sp_loss = sp_loss + F.softplus(p)
        sp_loss = -args.sparsity_weight * sp_loss

        # Optional: soft CTC penalty if we've drifted too far from baseline
        # (keeps thresholds from growing unboundedly at the cost of accuracy)
        loss = ctc_loss + sp_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(threshold_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # EMA of CTC loss for monitoring
        running_ctc = 0.95 * running_ctc + 0.05 * ctc_loss.item()

        # --- Logging ---
        if step % args.log_freq == 0:
            sparsities = output.all_sparsities or []
            avg_sp = sum(sparsities) / len(sparsities) if sparsities else 0.0
            t_vals = [F.softplus(p).item() for p in threshold_params]

            print(
                f"Step {step:5d}/{args.steps} | "
                f"CTC={ctc_loss.item():.4f} (ema={running_ctc:.4f}) | "
                f"sparsity={avg_sp:.1f}% | "
                f"t_mean={sum(t_vals)/len(t_vals):.4f} "
                f"t_min={min(t_vals):.4f} "
                f"t_max={max(t_vals):.4f} | "
                f"beta={beta:.1f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # --- Validation + checkpoint ---
        if step % args.val_freq == 0:
            val_ctc, val_sparsity = evaluate_baseline(model, val_loader, device)
            ctc_degradation = val_ctc - base_ctc

            print(f"\n  VAL step {step} | "
                  f"CTC={val_ctc:.4f} (Δ={ctc_degradation:+.4f} vs baseline) | "
                  f"sparsity={val_sparsity:.2f}%\n")

            # Warn if performance has degraded significantly
            if ctc_degradation > args.max_ctc_degradation:
                print(f"  WARNING: CTC degraded by {ctc_degradation:.4f} "
                      f"(limit={args.max_ctc_degradation}). "
                      f"Consider reducing --sparsity_weight.")

            # Save checkpoint (always save latest, also save best)
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ctc": val_ctc,
                "val_sparsity": val_sparsity,
                "base_ctc": base_ctc,
                "threshold_values": [F.softplus(p).item() for p in threshold_params],
                "args": vars(args),
            }
            torch.save(ckpt, output_dir / "latest.pt")

            if val_ctc < best_score:
                best_score = val_ctc
                torch.save(ckpt, output_dir / "best.pt")
                print(f"  New best saved (CTC={best_score:.4f})")

    # --- Final report ---
    print("\n=== Finetuning complete ===")
    final_ctc, final_sparsity = evaluate_baseline(model, val_loader, device, max_batches=50)
    print(f"  Baseline  CTC={base_ctc:.4f}  sparsity={base_sparsity:.2f}%")
    print(f"  Finetuned CTC={final_ctc:.4f}  sparsity={final_sparsity:.2f}%  "
          f"(ΔCTC={final_ctc - base_ctc:+.4f})")

    print("\nPer-layer final thresholds:")
    for name, p in get_threshold_params(model):
        print(f"  {name}: {F.softplus(p).item():.6f}")

    torch.save(model.state_dict(), output_dir / "final_thresholds.pt")
    print(f"\nSaved to {output_dir / 'final_thresholds.pt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output_dir",   default="./threshold_ckpts")
    p.add_argument("--cache_dir",    default="./cache")

    # Data
    p.add_argument("--dataset",      default="librispeech")
    p.add_argument("--max_samples",  type=int, default=-1)
    p.add_argument("--num_mfcc",     type=int, default=80)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--config_path", required=True)

    # Model architecture — must match pretraining config
    p.add_argument("--hidden_size",        type=int,   default=256)
    p.add_argument("--num_layers",         type=int,   default=6)
    p.add_argument("--state_size",         type=int,   default=128)
    p.add_argument("--expand",             type=int,   default=2)
    p.add_argument("--head_dim",           type=int,   default=64)
    p.add_argument("--chunk_size",         type=int,   default=256)
    p.add_argument("--initial_threshold",  type=float, default=0.1)

    # Finetuning
    p.add_argument("--steps",                type=int,   default=2000)
    p.add_argument("--lr",                   type=float, default=1e-2)
    p.add_argument("--sparsity_weight",      type=float, default=0.005)
    p.add_argument("--beta",                 type=float, default=10.0,
                   help="Initial sigmoid steepness")
    p.add_argument("--beta_max",             type=float, default=50.0,
                   help="Final sigmoid steepness (annealed)")
    p.add_argument("--max_ctc_degradation",  type=float, default=0.1,
                   help="Warn if val CTC rises more than this above baseline")

    # Logging
    p.add_argument("--log_freq",  type=int, default=50)
    p.add_argument("--val_freq",  type=int, default=500)

    return p.parse_args()


if __name__ == "__main__":
    finetune_thresholds(parse_args())