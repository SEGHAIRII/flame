#!/usr/bin/env python3
"""
Train a SentencePiece BPE tokenizer on LibriSpeech / People's Speech transcripts.
Saves the model + vocab to a known output directory, and prints the exact path
you need to pass as `spm_model_path` in your dataset class.

Usage:
    python train_sentencepiece.py                          # defaults
    python train_sentencepiece.py --vocab_size 1000
    python train_sentencepiece.py --dataset peoples_speech --vocab_size 2000
    python train_sentencepiece.py --output_dir ./tokenizers --split train.360
"""

import os
import argparse
import io
import soundfile as sf  # only used to confirm audio is readable; text-only here
from pathlib import Path

# ── deps ──────────────────────────────────────────────────────────────────────
try:
    import sentencepiece as spm
except ImportError:
    raise ImportError("Run:  pip install sentencepiece")

try:
    from datasets import load_dataset, Audio
except ImportError:
    raise ImportError("Run:  pip install datasets")


SAMPLE_RATE = 16000

os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["FSSPEC_HTTP_TIMEOUT"] = "3600"


# ── helpers ───────────────────────────────────────────────────────────────────

def iter_transcripts(dataset_name: str, split: str, max_samples: int):
    """Yield lower-cased transcripts from the chosen dataset."""

    if dataset_name == "librispeech":
        print(f"  Loading LibriSpeech split='{split}' (streaming) …")
        ds = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split=split,
            streaming=True,
        ).cast_column("audio", Audio(decode=False))

        for i, sample in enumerate(ds):
            if max_samples > 0 and i >= max_samples:
                break
            yield sample["text"].lower().strip()

    elif dataset_name == "peoples_speech":
        print(f"  Loading People's Speech split='{split}' (streaming) …")
        ds = load_dataset(
            "MLCommons/peoples_speech",
            "clean_sa",
            split=split,
            trust_remote_code=True,
            streaming=True,
        )
        for i, sample in enumerate(ds):
            if max_samples > 0 and i >= max_samples:
                break
            yield sample["text"].lower().strip()

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         "Choose 'librispeech' or 'peoples_speech'.")


def build_corpus(dataset_name: str, split: str, max_samples: int,
                 corpus_path: Path) -> int:
    """Stream transcripts → write one sentence per line to corpus_path."""
    print(f"\n[1/3] Building corpus file: {corpus_path}")
    n = 0
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for text in iter_transcripts(dataset_name, split, max_samples):
            if text:                        # skip empty lines
                fh.write(text + "\n")
                n += 1
                if n % 10_000 == 0:
                    print(f"  … {n:,} lines written", flush=True)
    print(f"  Done — {n:,} transcripts written to {corpus_path}")
    return n


def train_spm(corpus_path: Path, model_prefix: str, vocab_size: int):
    """Train SentencePiece BPE on the corpus."""
    print(f"\n[2/3] Training SentencePiece BPE  (vocab_size={vocab_size}) …")

    # We reserve index 0 for the CTC blank token by treating <blank> as a
    # user-defined symbol.  SentencePiece will place user symbols starting at
    # id=3 by default, so after training we do a +1 shift in the dataset class
    # (same as described in the integration instructions).
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size - 1,   # -1 because we add <blank> manually at 0
        model_type="bpe",
        character_coverage=1.0,      # 1.0 is correct for English / Latin script
        pad_id=-1,                   # no padding token
        unk_id=1,                    # unknown pieces get id 1 (shifted to 2 later)
        bos_id=-1,                   # no BOS
        eos_id=-1,                   # no EOS
        user_defined_symbols=["<blank>"],
        # keep whitespace info so the decoder can reconstruct words
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        input_sentence_size=5_000_000,   # cap memory usage during training
        shuffle_input_sentence=True,
    )


def summarise(model_prefix: str, vocab_size: int, output_dir: Path, corpus_path: Path):
    """Load the freshly trained model and print a summary."""
    model_file = Path(f"{model_prefix}.model")
    vocab_file  = Path(f"{model_prefix}.vocab")

    print(f"\n[3/3] Verifying model …")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    examples = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "automatic speech recognition",
        "i don't know",
    ]
    print("\n  Sample encodings (piece-level):")
    for ex in examples:
        pieces = sp.encode(ex, out_type=str)
        print(f"    '{ex}'  →  {pieces}")

    effective_vocab = sp.get_piece_size() + 1   # +1 for blank at 0
    print(f"\n  Effective vocab size (incl. <blank>): {effective_vocab}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  DONE")
    print("═" * 60)
    print(f"  Model file : {model_file.resolve()}")
    print(f"  Vocab file : {vocab_file.resolve()}")
    print(f"  Corpus file: {corpus_path.resolve()}  (safe to delete)")
    print(f"\n  Use these in your dataset class:")
    print(f"    spm_model_path = '{model_file.resolve()}'")
    print(f"\n  And in your SparseMamba2DeltaASRConfig:")
    print(f"    vocab_size = {effective_vocab}")
    print("═" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a SentencePiece BPE tokenizer on ASR transcripts."
    )
    p.add_argument("--dataset",     default="librispeech",
                   choices=["librispeech", "peoples_speech"],
                   help="Which dataset to pull transcripts from.")
    p.add_argument("--split",       default="train.360",
                   help="Dataset split.  LibriSpeech: train.100 / train.360 / train.500. "
                        "People's Speech: train.")
    p.add_argument("--vocab_size",  type=int, default=1000,
                   help="Total vocabulary size INCLUDING the <blank> token. "
                        "Recommended: 500, 1000 (default), 2000.")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Cap the number of transcripts used (-1 = all).")
    p.add_argument("--output_dir",  default="./tokenizers",
                   help="Directory where the model + vocab files are saved.")
    p.add_argument("--model_name",  default=None,
                   help="Base name for the .model / .vocab files.  "
                        "Defaults to  asr_bpe_<dataset>_<vocab_size>.")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or f"asr_bpe_{args.dataset}_{args.vocab_size}"
    model_prefix = str(output_dir / model_name)
    corpus_path  = output_dir / f"{model_name}_corpus.txt"

    print("=" * 60)
    print("  SentencePiece Tokenizer Training")
    print("=" * 60)
    print(f"  dataset    : {args.dataset}")
    print(f"  split      : {args.split}")
    print(f"  vocab_size : {args.vocab_size}  (incl. <blank>)")
    # print(f"  max_samples: {'all' if args.max_samples == -1 else args.max_samples:,}")
    max_samples_str = 'all' if args.max_samples == -1 else f"{args.max_samples:,}"
    print(f"  max_samples: {max_samples_str}")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  model_name : {model_name}")
    print("=" * 60)

    # 1. Build corpus
    n_lines = build_corpus(args.dataset, args.split, args.max_samples, corpus_path)
    if n_lines == 0:
        raise RuntimeError("Corpus is empty — check your dataset / split name.")

    # 2. Train
    train_spm(corpus_path, model_prefix, args.vocab_size)

    # 3. Summarise + print paths
    summarise(model_prefix, args.vocab_size, output_dir, corpus_path)


if __name__ == "__main__":
    main()