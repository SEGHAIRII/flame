#!/usr/bin/env python3
"""
Clean ASR evaluation script — mirrors the training pipeline exactly.

Sparsity (zero-rate of temporal differences on input and output projections)
is computed inside the model itself and returned as ``outputs.all_sparsities``
— a dict ``{"input": float, "output": float}`` for delta models, or None for
models that do not implement it.  This script just reads and logs those values.

Usage:
    # Standard model
    python eval_asr.py \
        --checkpoint path/to/best_model.pt \
        --config     path/to/config.json \
        --split      test \
        --batch_size 32

    # Delta model with quantization ON (activates sparsity metrics)
    python eval_asr.py \
        --checkpoint path/to/best_model.pt \
        --config     path/to/config.json \
        --model_name sparse_mamba2_delta_asr \
        --apply_qat  \
        --split      test
"""

import os
import json
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================================
# METRICS
# ============================================================================

def levenshtein_distance(seq1, seq2):
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    if not seq2:
        return len(seq1)
    prev = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        curr = [i + 1]
        for j, c2 in enumerate(seq2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def calculate_asr_metrics(predictions, targets, idx_to_char=None):
    metrics = {
        'cer': 0.0, 'wer': 0.0, 'exact_match': 0.0,
        'total_chars': 0, 'total_words': 0,
        'total_sequences': len(predictions),
    }
    if not predictions:
        return metrics

    tce = twe = tc = tw = em = 0
    for pred, target in zip(predictions, targets):
        pred   = pred.tolist()   if torch.is_tensor(pred)   else list(pred)
        target = target.tolist() if torch.is_tensor(target) else list(target)

        pc  = [p for p in pred   if p != 0]
        tc_ = [t for t in target if t != 0]

        tce += levenshtein_distance(pc, tc_)
        tc  += len(tc_)

        if idx_to_char:
            pred_text   = "".join([idx_to_char.get(t, "") for t in pc]).replace("▁", " ").strip()
            target_text = "".join([idx_to_char.get(t, "") for t in tc_]).replace("▁", " ").strip()
            pred_words, target_words = pred_text.split(), target_text.split()
        else:
            pred_words, target_words = pc, tc_

        twe += levenshtein_distance(pred_words, target_words)
        tw  += len(target_words)
        em  += (pc == tc_)

    metrics.update(
        cer=tce / max(tc, 1),
        wer=twe / max(tw, 1),
        exact_match=em / len(predictions),
        total_chars=tc,
        total_words=tw,
    )
    return metrics


def decode_ctc_greedy(logits, input_lengths, blank_id=0):
    argmax = torch.argmax(logits, dim=-1)
    results = []
    for b in range(argmax.size(0)):
        seq     = argmax[b, :input_lengths[b].item()].tolist()
        decoded, prev = [], None
        for t in seq:
            if t != blank_id and t != prev:
                decoded.append(t)
            prev = t
        results.append(decoded)
    return results


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config_path, checkpoint_path, device, model_name="sparse_mamba2_asr"):
    print(f"[1/4] Importing model class for '{model_name}' ...", flush=True)
    if model_name == "hgrn_asr":
        from sparse_mamba.custom_models.audio_models.hgrn import (
            HGRNASRConfig as CfgClass, HGRNASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_asr":
        from sparse_mamba.custom_models.audio_models.sparse_mamba2_ctc import (
            SparseMamba2ASRConfig as CfgClass, SparseMamba2ASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_delta_asr":
        from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
            SparseMamba2DeltaASRConfig as CfgClass, SparseMamba2DeltaASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_asr_noconv":
        from sparse_mamba.custom_models.audio_models.mamba2_noconv import (
            SparseMamba2ASRNoConvConfig as CfgClass, SparseMamba2ASRForCTC as ModelClass,
        )
    else:
        raise ValueError(
            f"Unknown --model_name '{model_name}'. "
            "Choose from: hgrn_asr, sparse_mamba2_asr, sparse_mamba2_delta_asr, sparse_mamba2_asr_noconv"
        )
    print("[1/4] Done.", flush=True)

    print(f"[2/4] Loading config from {config_path} ...", flush=True)
    if os.path.exists(config_path):
        config = CfgClass.from_pretrained(config_path)
    else:
        with open(config_path) as f:
            config = CfgClass(**json.load(f))
    print(f"[2/4] Config loaded — hidden={config.hidden_size}, "
          f"layers={config.num_hidden_layers}, vocab={config.vocab_size}", flush=True)

    print(f"[3/4] Building model on {device} ...", flush=True)
    model = ModelClass(config).to(device)
    print("[3/4] Done.", flush=True)

    print(f"[4/4] Loading checkpoint from {checkpoint_path} ...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"      Keys          : {list(ckpt.keys())}")
    print(f"      Saved at step : {ckpt.get('step', 'unknown')}")

    state_dict = ckpt['model_state_dict']
    sample_key = next(iter(state_dict))
    before = model.state_dict()[sample_key].flatten()[:3].clone()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        if all("_threshold_raw" in k for k in missing):
            raise RuntimeError(
                f"Missing threshold params: {missing}. "
                "Load a QAT checkpoint (not pretraining) for delta models."
            )
        raise RuntimeError(f"Missing keys: {missing}")

    after = model.state_dict()[sample_key].flatten()[:3]
    assert not torch.allclose(before, after), "Weights did NOT change — checkpoint not loaded!"
    model.eval()
    print("[4/4] Checkpoint loaded.", flush=True)
    return model, config


# ============================================================================
# DATALOADER
# ============================================================================

def build_test_loader(cache_dir, split, batch_size, max_samples, num_workers):
    from sparse_mamba.custom_dataloaders.preprocess_ctc import CachedASRDataset, collate_fn_ctc

    split_map = {
        'train': 'train.960', 'val': 'validation',
        'validation': 'validation', 'test': 'test',
    }
    cache_split = split_map.get(split, split)
    print(f"Loading cached features from {cache_dir}/{cache_split} ...")
    ds = CachedASRDataset(cache_dir, cache_split, max_samples)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_ctc, num_workers=num_workers, pin_memory=True,
    )
    loader.idx_to_char = ds.idx_to_char
    loader.char_to_idx = ds.char_to_idx
    print(f"Dataset: {len(ds)} samples, vocab={len(ds.char_to_idx)}")
    return loader


# ============================================================================
# EVALUATE
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, max_batches=-1, verbose=True,
             apply_qat_quantization=False):
    """
    Evaluate the model.

    ``outputs.all_sparsities`` is expected to be a dict
    ``{"input": float, "output": float}`` when the model supports it (delta
    models with ``return_activation_sparsity=True``), or None otherwise.
    All sparsity computation is done inside the model; this function only
    accumulates and reports the results.

    Parameters
    ----------
    apply_qat_quantization : bool
        Passed through to the model when True so that discrete tensors are
        populated and temporal-diff sparsity is computed over quantised values.
        Has no effect on models that do not support the kwarg.
    """
    model.eval()

    all_preds           = []
    all_targets         = []
    total_loss          = 0.0
    sparsity_input_acc  = []
    sparsity_output_acc = []

    # Try once with the kwarg; fall back silently if unsupported.
    _qat_kwarg = {"apply_qat_quantization": True} if apply_qat_quantization else {}

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch['targets'].to(device, dtype=torch.long)
        input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        try:
            outputs = model(
                input_ids=inputs, labels=labels,
                input_lengths=input_lengths, target_lengths=target_lengths,
                **_qat_kwarg,
            )
        except TypeError:
            # Model does not accept apply_qat_quantization — drop it for all future batches
            _qat_kwarg = {}
            outputs = model(
                input_ids=inputs, labels=labels,
                input_lengths=input_lengths, target_lengths=target_lengths,
            )

        if outputs.loss is not None:
            total_loss += outputs.loss.item()

        # Sparsity is computed inside the model and returned as a dict or None
        sp = getattr(outputs, 'all_sparsities', None)
        if isinstance(sp, dict):
            if sp.get('input')  is not None: sparsity_input_acc.append(sp['input'])
            if sp.get('output') is not None: sparsity_output_acc.append(sp['output'])

        # Use subsampled lengths for direct logit decoding.
        # model.decode() runs the frontend internally so always give it the original lengths.
        effective_lengths = input_lengths
        if hasattr(outputs, 'output_lengths') and outputs.output_lengths is not None:
            effective_lengths = outputs.output_lengths

        if hasattr(model, 'decode'):
            preds = model.decode(inputs, input_lengths, use_beam_search=False)
        else:
            blank_id = getattr(model, 'blank_id', 0)
            preds = decode_ctc_greedy(
                F.log_softmax(outputs.logits.float(), dim=-1),
                effective_lengths, blank_id=blank_id,
            )

        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

        if verbose and (i + 1) % 10 == 0:
            print(f"  batch {i+1} done ...")

    # Example predictions
    if verbose:
        idx_to_char = loader.idx_to_char
        print("\n--- Example Predictions ---")
        for k in range(min(5, len(all_preds))):
            pred_text   = "".join([idx_to_char.get(t, "?") for t in all_preds[k]]).replace("▁", " ")
            target_text = "".join([idx_to_char.get(t, "?") for t in all_targets[k]]).replace("▁", " ")
            print(f"  REF : {target_text}")
            print(f"  HYP : {pred_text}")
            print()

    n_batches = max(i + 1, 1)
    metrics = calculate_asr_metrics(all_preds, all_targets, loader.idx_to_char)
    metrics['loss'] = total_loss / n_batches
    if sparsity_input_acc:
        metrics['sparsity_input']  = sum(sparsity_input_acc)  / len(sparsity_input_acc)
    if sparsity_output_acc:
        metrics['sparsity_output'] = sum(sparsity_output_acc) / len(sparsity_output_acc)

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "validation", "test"])
    p.add_argument("--cache_dir",   required=True)
    p.add_argument("--dataset",     default="librispeech",
                   choices=["librispeech", "peoples_speech"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  default="eval_results")
    p.add_argument("--device",      default=None)
    p.add_argument("--model_name",  default="sparse_mamba2_asr",
                   choices=["hgrn_asr", "sparse_mamba2_asr",
                             "sparse_mamba2_delta_asr", "sparse_mamba2_asr_noconv"])
    p.add_argument("--apply_qat", action="store_true",
                   help="Enable quantization for delta models. Populates discrete "
                        "tensors so the model can compute temporal-diff sparsity.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model, config = load_model(args.config, args.checkpoint, device, args.model_name)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.2f}M", flush=True)

    print("\n[Step 2] Building dataloader ...", flush=True)
    loader = build_test_loader(
        cache_dir=args.cache_dir, split=args.split,
        batch_size=args.batch_size, max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print("\n[Step 3] Sanity-checking first batch ...", flush=True)
    sample = next(iter(loader))['features'].to(device, dtype=torch.float32).transpose(1, 2)
    print(f"Feature shape : {sample.shape}")
    print(f"Feature mean  : {sample.mean():.4f}")
    print(f"Feature std   : {sample.std():.4f}", flush=True)

    print(f"\n[Step 4] Evaluating {args.dataset} {args.split} "
          f"(quantization={'ON' if args.apply_qat else 'OFF'}) ...", flush=True)
    metrics = evaluate(
        model, loader, device,
        max_batches=args.max_batches,
        verbose=True,
        apply_qat_quantization=args.apply_qat,
    )

    print("\n" + "=" * 60)
    print(f"  Loss          : {metrics['loss']:.4f}")
    print(f"  CER           : {metrics['cer']*100:.2f}%")
    print(f"  WER           : {metrics['wer']*100:.2f}%")
    print(f"  Exact Match   : {metrics['exact_match']*100:.2f}%")
    print(f"  Total seqs    : {metrics['total_sequences']}")
    if 'sparsity_input' in metrics:
        print(f"  Sparsity in   : {metrics['sparsity_input']:.2f}%")
        print(f"  Sparsity out  : {metrics.get('sparsity_output', 0):.2f}%")
    print("=" * 60)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_{args.split}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "checkpoint":  args.checkpoint,
                "config":      args.config,
                "model_name":  args.model_name,
                "split":       args.split,
                "dataset":     args.dataset,
                "apply_qat":   args.apply_qat,
                "step":        torch.load(args.checkpoint, map_location="cpu").get("step", 0),
            },
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()
                        if isinstance(v, (int, float))},
        }, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()