#!/usr/bin/env python3
"""
Standalone evaluation script for SparseMamba2 Delta ASR model.
Usage:
    python eval_asr.py \
        --checkpoint ../sparse_mamba/exp/sparse_mamba2_asr/checkpoints/best_model.pt \
        --config ../sparse_mamba/configs/sparse_mamba2_ctc.json \
        --dataset librispeech \
        --split test \
        --beam_size 1 \
        --batch_size 32 \
        --return_sparsity \
        --output_dir results/
"""

import os
import json
import argparse
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict


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
        'cer': 0.0, 'wer': 0.0, 'exact_match': 0.0, 'token_accuracy': 0.0,
        'total_chars': 0, 'total_words': 0, 'total_sequences': len(predictions)
    }
    if not predictions:
        return metrics

    tce = twe = tc = tw = em = tk_c = tk_t = 0
    for pred, target in zip(predictions, targets):
        pred   = pred.tolist()   if torch.is_tensor(pred)   else list(pred)
        target = target.tolist() if torch.is_tensor(target) else list(target)
        pc  = [p for p in pred   if p != 0]
        tc_ = [t for t in target if t != 0]
        tce += levenshtein_distance(pc, tc_)
        tc  += len(tc_)
        twe += levenshtein_distance(pred, target)
        tw  += len(target)
        em  += pred == target
        ml   = min(len(pred), len(target))
        tk_c += sum(pred[i] == target[i] for i in range(ml))
        tk_t += max(len(pred), len(target))

    metrics.update(
        cer=tce / max(tc, 1),
        wer=twe / max(tw, 1),
        exact_match=em / len(predictions),
        token_accuracy=tk_c / max(tk_t, 1),
        total_chars=tc,
        total_words=tw,
    )
    return metrics


def decode_predictions_ctc(logits, input_lengths, blank_id=0):
    preds = torch.argmax(logits, dim=-1) if logits.dim() == 3 else logits
    results = []
    for b in range(preds.size(0)):
        seq = preds[b, :input_lengths[b].item()].tolist()
        decoded, prev = [], None
        for t in seq:
            if t != blank_id and t != prev:
                decoded.append(t)
            prev = t
        results.append(decoded)
    return results


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def peek_checkpoint(checkpoint_path: str):
    """Peek at checkpoint to auto-detect architecture."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dicts' in checkpoint:
        state_dict = checkpoint['model_state_dicts'][0]
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # infer num_layers
    layer_indices = set()
    for k in state_dict.keys():
        parts = k.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    num_layers = max(layer_indices) + 1 if layer_indices else None

    # infer hidden_size from ctc_head weight: [vocab_size, hidden_size]
    hidden_size = None
    for k, v in state_dict.items():
        if 'ctc_head.weight' in k:
            hidden_size = v.shape[1]
            break

    step = checkpoint.get('step', 0)
    return num_layers, hidden_size, step, state_dict


def load_checkpoint(model, state_dict):
    model.load_state_dict(state_dict, strict=True)
    return model


# ============================================================================
# DATALOADER
# ============================================================================

def build_eval_dataloader(dataset: str, split: str, batch_size: int,
                           num_mfcc: int, max_samples: int, num_workers: int):
    from sparse_mamba.custom_dataloaders.librosa import create_librosa_raw_classification_dataset

    print(f"Loading {dataset} {split} split...")
    train_loader, val_loader, test_loader, n_classes, seq_len, input_dim, char_to_idx, idx_to_char = \
        create_librosa_raw_classification_dataset(
            bsz=batch_size,
            max_samples=max_samples,
            num_mfcc=num_mfcc,
            dataset=dataset,
            num_workers=num_workers,
        )

    split_map = {
        'train':      train_loader,
        'val':        val_loader,
        'validation': val_loader,
        'test':       test_loader,
    }
    loader = split_map.get(split, test_loader)
    return loader, n_classes, input_dim, char_to_idx, idx_to_char


# ============================================================================
# EVALUATE
# ============================================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    idx_to_char: Dict[int, str],
    use_beam_search: bool = False,
    max_batches: int = -1,
    verbose: bool = True,
):
    model.eval()
    total_loss     = 0.0
    all_preds      = []
    all_targets    = []
    all_sparsities = []
    num_batches    = 0

    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break

        features       = batch['features'].to(device, dtype=torch.float32)
        inputs         = features.transpose(1, 2) if features.dim() == 3 else features
        labels         = batch.get('targets', batch.get('text_tokens')).to(device, dtype=torch.long)
        input_lengths  = batch.get('feature_lengths', batch.get('lengths')).to(device, dtype=torch.long)
        target_lengths = batch.get('target_lengths',
                            torch.tensor([labels.shape[0]], device=device, dtype=torch.long))

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=inputs,
                labels=labels,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )

        if outputs.loss is not None:
            total_loss += outputs.loss.item()

        # collect activation sparsity
        sparsities = getattr(outputs, 'activation_sparsities', [])
        if sparsities:
            all_sparsities.extend(sparsities)

        # decode
        if hasattr(model, 'decode'):
            preds = model.decode(inputs, input_lengths, use_beam_search=use_beam_search)
        else:
            preds = decode_predictions_ctc(outputs.logits, input_lengths)

        # collect targets per sequence
        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

        num_batches += 1
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} batches...")

    # ASR metrics
    metrics = calculate_asr_metrics(all_preds, all_targets, idx_to_char)
    metrics['loss'] = total_loss / max(num_batches, 1)

    # activation sparsity metrics
    if all_sparsities:
        metrics['act_sparsity_mean'] = sum(all_sparsities) / len(all_sparsities)
        metrics['act_sparsity_min']  = min(all_sparsities)
        metrics['act_sparsity_max']  = max(all_sparsities)
        metrics['act_sparsity_std']  = float(torch.tensor(all_sparsities).std().item())
        # per-layer average
        num_layers = model.config.num_hidden_layers
        if len(all_sparsities) % num_layers == 0:
            layer_sparsities = {}
            for l in range(num_layers):
                vals = all_sparsities[l::num_layers]
                layer_sparsities[f'layer_{l}'] = sum(vals) / len(vals)
            metrics['act_sparsity_per_layer'] = layer_sparsities

    # example predictions
    if verbose and idx_to_char:
        print("\n--- Example Predictions ---")
        for k in range(min(5, len(all_preds))):
            pred_text   = "".join([idx_to_char.get(t, "?") for t in all_preds[k]])
            target_text = "".join([idx_to_char.get(t, "?") for t in all_targets[k]])
            print(f"  REF : {target_text}")
            print(f"  HYP : {pred_text}")
            print()

    return metrics


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(metrics: dict, args, step: int, num_layers: int, hidden_size: int):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    decoding  = f"beam{args.beam_size}" if args.beam_size > 1 else "greedy"
    filename  = f"eval_{args.split}_{decoding}_{timestamp}.json"
    filepath  = output_dir / filename

    results = {
        "metadata": {
            "timestamp":       timestamp,
            "checkpoint":      args.checkpoint,
            "dataset":         args.dataset,
            "split":           args.split,
            "decoding":        decoding,
            "beam_size":       args.beam_size,
            "batch_size":      args.batch_size,
            "step":            step,
            "num_layers":      num_layers,
            "hidden_size":     hidden_size,
            "return_sparsity": args.return_sparsity,
        },
        "metrics": {
            "loss":            round(metrics['loss'], 6),
            "cer":             round(metrics['cer'], 6),
            "cer_pct":         round(metrics['cer'] * 100, 4),
            "wer":             round(metrics['wer'], 6),
            "wer_pct":         round(metrics['wer'] * 100, 4),
            "exact_match":     round(metrics['exact_match'], 6),
            "token_accuracy":  round(metrics['token_accuracy'], 6),
            "total_sequences": metrics['total_sequences'],
            "total_chars":     metrics['total_chars'],
            "total_words":     metrics['total_words'],
        },
    }

    if 'act_sparsity_mean' in metrics:
        results["activation_sparsity"] = {
            "mean": round(metrics['act_sparsity_mean'], 6),
            "min":  round(metrics['act_sparsity_min'],  6),
            "max":  round(metrics['act_sparsity_max'],  6),
            "std":  round(metrics['act_sparsity_std'],  6),
        }
        if 'act_sparsity_per_layer' in metrics:
            results["activation_sparsity"]["per_layer"] = {
                k: round(v, 6) for k, v in metrics['act_sparsity_per_layer'].items()
            }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SparseMamba2 Delta ASR model")

    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--dataset",         type=str, default="librispeech",
                        choices=["librispeech", "peoples_speech"])
    parser.add_argument("--split",           type=str, default="test",
                        choices=["train", "val", "validation", "test"])
    parser.add_argument("--beam_size",       type=int, default=1)
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--max_batches",     type=int, default=-1)
    parser.add_argument("--max_samples",     type=int, default=-1)
    parser.add_argument("--num_mfcc",        type=int, default=80)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--device",          type=str, default=None)
    parser.add_argument("--state_size",      type=int, default=64)
    parser.add_argument("--expand",          type=int, default=2)
    parser.add_argument("--head_dim",        type=int, default=64)
    parser.add_argument("--chunk_size",      type=int, default=256)
    parser.add_argument("--return_sparsity", action="store_true",
                        help="Compute and log activation sparsity per layer")
    parser.add_argument("--output_dir",      type=str, default="eval_results",
                        help="Directory to save JSON results")

    return parser.parse_args()


def main():
    args = parse_args()

    # device
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # peek at checkpoint to auto-detect architecture
    print(f"Peeking at checkpoint: {args.checkpoint}")
    num_layers, hidden_size, step, state_dict = peek_checkpoint(args.checkpoint)
    print(f"Auto-detected: num_layers={num_layers}, hidden_size={hidden_size}, step={step}")

    # dataloader
    loader, n_classes, input_dim, char_to_idx, idx_to_char = build_eval_dataloader(
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        num_mfcc=args.num_mfcc,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {n_classes} classes, input_dim={input_dim}")

    # build model with auto-detected architecture
    from sparse_mamba.custom_models.audio_models.delta_mamba2 import (
        SparseMamba2DeltaASRConfig, SparseMamba2DeltaASRForCTC
    )
    config = SparseMamba2DeltaASRConfig(
        input_size=input_dim,
        vocab_size=n_classes,
        blank_id=0,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        state_size=args.state_size,
        expand=args.expand,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        return_activation_sparsity=args.return_sparsity,
        rms_norm=True,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        norm_eps=1e-5,
        residual_in_fp32=True,
    )

    model = SparseMamba2DeltaASRForCTC(config).to(device)

    # load checkpoint
    model = load_checkpoint(model, state_dict)
    print(f"Checkpoint loaded (step {step})")

    # decoding setup
    use_beam_search = args.beam_size > 1
    if use_beam_search:
        print(f"Building torchaudio CTCDecoder with beam_size={args.beam_size}")
        model.build_decoder(idx_to_char, beam_size=args.beam_size)
    else:
        print("Using greedy decoding")

    model.eval()

    # evaluate
    print(f"\nEvaluating on {args.dataset} {args.split} split...")
    metrics = evaluate(
        model=model,
        dataloader=loader,
        device=device,
        idx_to_char=idx_to_char,
        use_beam_search=use_beam_search,
        max_batches=args.max_batches,
        verbose=True,
    )

    # print results
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS (step {step})")
    print("=" * 50)
    print(f"Loss          : {metrics['loss']:.4f}")
    print(f"CER           : {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"WER           : {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Exact Match   : {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"Token Accuracy: {metrics['token_accuracy']:.4f} ({metrics['token_accuracy']*100:.2f}%)")
    print(f"Sequences     : {metrics['total_sequences']}")
    if 'act_sparsity_mean' in metrics:
        print(f"\nActivation Sparsity:")
        print(f"  Mean : {metrics['act_sparsity_mean']:.4f}")
        print(f"  Min  : {metrics['act_sparsity_min']:.4f}")
        print(f"  Max  : {metrics['act_sparsity_max']:.4f}")
        print(f"  Std  : {metrics['act_sparsity_std']:.4f}")
        if 'act_sparsity_per_layer' in metrics:
            print(f"  Per layer:")
            for layer, val in metrics['act_sparsity_per_layer'].items():
                print(f"    {layer}: {val:.4f}")
    print("=" * 50)

    # save to json
    save_results(metrics, args, step, num_layers, hidden_size)

    return metrics


if __name__ == "__main__":
    main()