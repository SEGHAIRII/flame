#!/usr/bin/env python3
"""
Clean ASR evaluation script — mirrors training pipeline exactly.

Usage:
    python eval_asr.py \
        --checkpoint path/to/best_model.pt \
        --config     path/to/sparse_mamba2_relu_ctc.json \
        --split      test \
        --batch_size 32
"""

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
        'total_sequences': len(predictions)
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
            pred_words   = pred_text.split()
            target_words = target_text.split()
        else:
            pred_words   = pc
            target_words = tc_

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
    """Standard CTC greedy decode — collapse repeats then remove blanks."""
    argmax = torch.argmax(logits, dim=-1)   # (B, T)
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
# CHECKPOINT
# ============================================================================

def load_model(config_path: str, checkpoint_path: str, device: torch.device,
               model_name: str = "sparse_mamba2_asr"):
    if model_name == "hgrn_asr":
        from sparse_mamba.custom_models.audio_models.hgrn import (
            HGRNASRConfig as CfgClass,
            HGRNASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_asr":
        from sparse_mamba.custom_models.audio_models.sparse_mamba2_ctc import (
            SparseMamba2ASRConfig as CfgClass,
            SparseMamba2ASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_delta_asr":
        from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
            SparseMamba2DeltaASRConfig as CfgClass,
            SparseMamba2DeltaASRForCTC as ModelClass,
        )
    elif model_name == "sparse_mamba2_asr_noconv":
        from sparse_mamba.custom_models.audio_models.mamba2_noconv import (
            SparseMamba2ASRConfig as CfgClass,
            SparseMamba2ASRForCTC as ModelClass,
        )
    else:
        raise ValueError(
            f"Unknown --model_name '{model_name}'. "
            f"Choose from: hgrn_asr, sparse_mamba2_asr, sparse_mamba2_delta_asr, sparse_mamba2_asr_noconv"
        )

    # 1. Load config with same path as training registry
    if os.path.exists(config_path):
        config = CfgClass.from_pretrained(config_path)
    else:
        with open(config_path) as f:
            cfg = json.load(f)
        config = CfgClass(**cfg)
    print(f"Config loaded — hidden={config.hidden_size}, "
          f"layers={config.num_hidden_layers}, vocab={config.vocab_size}")

    # 2. Build model
    model = ModelClass(config).to(device)

    # 3. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys : {list(ckpt.keys())}")
    print(f"Saved at step   : {ckpt.get('step', 'unknown')}")
    print(f"Best val CER    : {ckpt.get('train_state', {}).get('best_val_cer', 'unknown')}")

    state_dict = ckpt['model_state_dict']
    print("First 5 keys in checkpoint:", list(state_dict.keys())[:5])
    print("First 5 keys in model:", list(model.state_dict().keys())[:5])

    # Verify a weight actually changes after loading
    sample_key = next(iter(state_dict))
    before = model.state_dict()[sample_key].flatten()[:3].clone()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {unexpected}. "
            f"This usually means --model_name does not match the training model variant."
        )
    if missing:
        if all("_threshold_raw" in k for k in missing):
            raise RuntimeError(
                f"Missing threshold params: {missing}. "
                f"This usually means a non-delta checkpoint is being loaded into "
                f"a delta model. Check --model_name."
            )
        raise RuntimeError(f"Missing keys: {missing}")

    after = model.state_dict()[sample_key].flatten()[:3]
    print(f"Weight check — before: {before.tolist()}")
    print(f"Weight check — after : {after.tolist()}")
    assert not torch.allclose(before, after), "Weights did NOT change — checkpoint not loaded!"

    model.eval()
    return model, config


# ============================================================================
# DATALOADER  — uses exact same code path as training
# ============================================================================

def build_test_loader(dataset: str, split: str, batch_size: int,
                      num_mfcc: int, max_samples: int, num_workers: int,
                      spm_model_path: Optional[str] = None,
                      streaming: bool = False):
    """
    Uses LibriSpeechASRDataset + collate_fn_ctc — identical to training.
    This guarantees feature extraction is byte-for-byte the same.
    """
    from sparse_mamba.custom_dataloaders.preprocess_ctc import (
        LibriSpeechASRDataset, collate_fn_ctc
    )

    split_map = {
        'train':      'train.960',
        'val':        'validation',
        'validation': 'validation',
        'test':       'test',
    }
    hf_split = split_map.get(split, split)

    print(f"Loading {dataset} / {hf_split} ...")
    ds = LibriSpeechASRDataset(
        split=hf_split,
        max_samples=max_samples,
        num_mfcc=num_mfcc,
        dataset=dataset,
        spm_model_path=spm_model_path,
        streaming=streaming,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Attach metadata
    loader.idx_to_char = ds.idx_to_char
    loader.char_to_idx = ds.char_to_idx
    print(f"Dataset: {len(ds)} samples, vocab={len(ds.char_to_idx)}")
    return loader


# ============================================================================
# EVALUATE
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, max_batches=-1, verbose=True):
    model.eval()

    all_preds      = []
    all_targets    = []
    all_sparsities = []
    total_loss     = 0.0
    idx_to_char    = loader.idx_to_char

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        # Exactly the same as training's train_step input preparation
        inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
        labels         = batch['targets'].to(device, dtype=torch.long)
        input_lengths  = batch['feature_lengths'].to(device, dtype=torch.long)
        target_lengths = batch['target_lengths'].to(device, dtype=torch.long)

        outputs = model(
            input_ids=inputs,
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        if outputs.loss is not None:
            total_loss += outputs.loss.item()

        if outputs.all_sparsities:
            all_sparsities.extend(outputs.all_sparsities)

        # Match training-time evaluation path
        if hasattr(model, 'decode'):
            preds = model.decode(inputs, input_lengths, use_beam_search=False)
        else:
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
            preds = decode_ctc_greedy(log_probs, input_lengths, blank_id=0)

        # Recover per-sequence targets from flat CTC target tensor
        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

        if verbose and (i + 1) % 10 == 0:
            print(f"  batch {i+1} done ...")

    # Show examples
    if verbose:
        print("\n--- Example Predictions ---")
        for k in range(min(5, len(all_preds))):
            pred_text   = "".join([idx_to_char.get(t, "?") for t in all_preds[k]]).replace("▁", " ")
            target_text = "".join([idx_to_char.get(t, "?") for t in all_targets[k]]).replace("▁", " ")
            print(f"  REF : {target_text}")
            print(f"  HYP : {pred_text}")
            print()

    n_batches = max(i + 1, 1)
    metrics = calculate_asr_metrics(all_preds, all_targets, idx_to_char)
    metrics['loss'] = total_loss / n_batches

    if all_sparsities:
        metrics['act_sparsity_mean'] = sum(all_sparsities) / len(all_sparsities)
        metrics['act_sparsity_min']  = min(all_sparsities)
        metrics['act_sparsity_max']  = max(all_sparsities)

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
    p.add_argument("--dataset",     default="librispeech",
                   choices=["librispeech", "peoples_speech"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_mfcc",    type=int, default=None,
                   help="Feature dimension. Defaults to config.input_size when not provided.")
    p.add_argument("--spm_model_path", default=None,
                   help="SentencePiece model path used during training.")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming dataset mode (should match training setting).")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  default="eval_results")
    p.add_argument("--device",      default=None)
    p.add_argument("--model_name", default="sparse_mamba2_asr",
                   choices=[
                       "hgrn_asr",
                       "sparse_mamba2_asr",
                       "sparse_mamba2_delta_asr",
                       "sparse_mamba2_asr_noconv",
                   ],
                   help="Must match the model.name used during training.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model
    model, config = load_model(args.config, args.checkpoint, device, args.model_name)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.2f}M")

    # 2. Build dataloader (same code path as training)
    effective_num_mfcc = args.num_mfcc if args.num_mfcc is not None else int(config.input_size)
    loader = build_test_loader(
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        num_mfcc=effective_num_mfcc,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        spm_model_path=args.spm_model_path,
        streaming=args.streaming,
    )

    # 3. Sanity-check: features should look reasonable
    batch = next(iter(loader))
    sample = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
    print(f"\nFeature shape : {sample.shape}")
    print(f"Feature mean  : {sample.mean():.4f}  (training used ~0)")
    print(f"Feature std   : {sample.std():.4f}   (training used ~0.5-1)")

    # 4. Evaluate
    # print(f"\nEvaluating {args.dataset} {args.split} ...")
    # metrics = evaluate(model, loader, device,
    #                    max_batches=args.max_batches, verbose=True)


    # 4. Evaluate
    print(f"\nEvaluating {args.dataset} {args.split} ...")
    metrics = evaluate(model, loader, device,
                       max_batches=args.max_batches, verbose=True)
    # 5. Print results
    print("\n" + "=" * 55)
    print(f"  Loss          : {metrics['loss']:.4f}")
    print(f"  CER           : {metrics['cer']*100:.2f}%")
    print(f"  WER           : {metrics['wer']*100:.2f}%")
    print(f"  Exact Match   : {metrics['exact_match']*100:.2f}%")
    print(f"  Total seqs    : {metrics['total_sequences']}")
    print(f"  Total chars   : {metrics['total_chars']}")
    if 'act_sparsity_mean' in metrics:
        print(f"  Sparsity mean : {metrics['act_sparsity_mean']:.4f}")
    print("=" * 55)

    # 6. Save JSON
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"eval_{args.split}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "checkpoint": args.checkpoint,
                "config":     args.config,
                "model_name": args.model_name,
                "split":      args.split,
                "dataset":    args.dataset,
                "step":       torch.load(args.checkpoint,
                               map_location="cpu").get("step", 0),
            },
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()
                        if isinstance(v, (int, float))}
        }, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()