#!/usr/bin/env python3
"""
Standalone evaluation script for SparseMamba2 ASR model.
Usage:
    python eval_asr.py \
        --checkpoint ../sparse_mamba/exp/sparse_mamba2_asr/checkpoints/best_model.pt \
        --config ../sparse_mamba/configs/sparse_mamba2_ctc.json \
        --dataset librispeech \
        --split test \
        --beam_size 10 \
        --batch_size 32 \
        --max_batches -1
"""

import os
import argparse
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

def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # handle different checkpoint formats
    if 'model_state_dicts' in checkpoint:
        # ASRCheckpointManager format
        state_dict = checkpoint['model_state_dicts'][0]
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # raw state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    step = checkpoint.get('step', checkpoint.get('train_state', {}).get('step', 0))
    print(f"Loaded checkpoint at step {step}")
    return model, step


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
        'train': train_loader,
        'val':   val_loader,
        'validation': val_loader,
        'test':  test_loader,
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
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    all_sparsities = []
    num_batches = 0

    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break


        features = batch['features'].to(device, dtype=torch.float32)
        inputs = features.transpose(1, 2) if features.dim() == 3 else features
        # inputs         = batch['features'].to(device, dtype=torch.float32).transpose(1, 2)
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

        # collect activation sparsity if available
        sparsities = getattr(outputs, 'activation_sparsities', [])
        if sparsities:
            all_sparsities.extend(sparsities)

        # decode
        if hasattr(model, 'decode'):
            preds = model.decode(inputs, input_lengths, use_beam_search=use_beam_search)
        else:
            preds = decode_predictions_ctc(outputs.logits, input_lengths)

        # collect targets
        start = 0
        for j in range(len(preds)):
            t = target_lengths[j].item()
            all_targets.append(labels[start:start + t].tolist())
            start += t
        all_preds.extend(preds)

        num_batches += 1
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} batches...")

    # compute metrics
    metrics = calculate_asr_metrics(all_preds, all_targets, idx_to_char)
    metrics['loss'] = total_loss / max(num_batches, 1)

    # add sparsity
    if all_sparsities:
        metrics['act_sparsity_mean'] = sum(all_sparsities) / len(all_sparsities)
        metrics['act_sparsity_min']  = min(all_sparsities)
        metrics['act_sparsity_max']  = max(all_sparsities)

    # show some example predictions
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
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SparseMamba2 ASR model")

    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to checkpoint file (best_model.pt or checkpoint_step_N.pt)")
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--dataset",     type=str, default="librispeech",
                        choices=["librispeech", "peoples_speech"],
                        help="Dataset to evaluate on")
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "validation", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--beam_size",   type=int, default=1,
                        help="Beam size (1 = greedy, >1 = beam search)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Max batches to evaluate (-1 = all)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max dataset samples to load (-1 = all)")
    parser.add_argument("--num_mfcc",    type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not set.")
    parser.add_argument("--hidden_size",      type=int, default=512)
    parser.add_argument("--num_layers",       type=int, default=6)
    parser.add_argument("--state_size",       type=int, default=64)
    parser.add_argument("--expand",           type=int, default=2)
    parser.add_argument("--head_dim",         type=int, default=64)
    parser.add_argument("--chunk_size",       type=int, default=256)
    parser.add_argument("--return_sparsity",  action="store_true",
                        help="Return and log activation sparsity")

    return parser.parse_args()


def main():
    args = parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # build model
    from sparse_mamba.custom_models.audio_models.sparse_mamba2_ctc import (
        SparseMamba2ASRConfig, SparseMamba2ASRForCTC
    )

    config = SparseMamba2ASRConfig(
        input_size=input_dim,
        vocab_size=n_classes,
        blank_id=0,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
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

    model = SparseMamba2ASRForCTC(config).to(device)

    # set beam size
    use_beam_search = args.beam_size > 1
    if use_beam_search:
        # model.decoder.beam_size = args.beam_size
        model.build_decoder(idx_to_char, beam_size=args.beam_size)

        print(f"Using beam search with beam_size={args.beam_size}")
    else:
        print("Using greedy decoding")

    # load checkpoint
    model, step = load_checkpoint(model, args.checkpoint, device)
    
    if use_beam_search:
        print(f"Building torchaudio CTCDecoder with beam_size={args.beam_size}")
        model.build_decoder(idx_to_char, beam_size=args.beam_size)

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
        print(f"Act Sparsity  : {metrics['act_sparsity_mean']:.2f}% "
              f"(min={metrics['act_sparsity_min']:.2f}%, max={metrics['act_sparsity_max']:.2f}%)")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    main()