# -*- coding: utf-8 -*-

import argparse
import io
import sys
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

# Make the script runnable directly (without pre-setting PYTHONPATH).
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "flame"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import fla  # noqa: F401
import torch
import torch.serialization
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sparse_mamba import custom_models  # noqa: F401


def _looks_like_state_dict(obj: Mapping) -> bool:
    if not obj:
        return False
    return all(torch.is_tensor(v) for v in obj.values())


def _extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, Mapping):
        raise ValueError(
            f"Unsupported checkpoint type: {type(checkpoint_obj)}. Expected a dict-like object."
        )

    if "model_state_dict" in checkpoint_obj and isinstance(
        checkpoint_obj["model_state_dict"], Mapping
    ):
        return checkpoint_obj["model_state_dict"]

    if "model_state_dicts" in checkpoint_obj:
        state_dicts = checkpoint_obj["model_state_dicts"]
        if isinstance(state_dicts, (list, tuple)) and state_dicts and isinstance(
            state_dicts[0], Mapping
        ):
            return state_dicts[0]

    if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], Mapping):
        return checkpoint_obj["model"]

    if _looks_like_state_dict(checkpoint_obj):
        return checkpoint_obj

    raise ValueError(
        "Could not find model weights in checkpoint. Expected one of: "
        "'model_state_dict', 'model_state_dicts[0]', 'model', or a raw state_dict."
    )


@torch.inference_mode()
def convert_pt_to_hf(
    checkpoint_path: str,
    config_path: str,
    tokenizer_path: str,
    output_dir: str,
    safe_serialization: bool = False,
    strict: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading config from: {config_path}")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.save_pretrained(output_path)

    print(f"[2/4] Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"[3/4] Loading checkpoint from: {checkpoint_path}")
    # Needed for some checkpoints that serialize timedelta/BytesIO in metadata.
    torch.serialization.add_safe_globals([timedelta, io.BytesIO])
    checkpoint_obj = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = _extract_state_dict(checkpoint_obj)

    model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")

    print(f"[4/4] Saving HF checkpoint to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=safe_serialization)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a plain .pt checkpoint into Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint (final_model.pt or training checkpoint).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config path/ID (e.g. sparse_mamba/configs/delta_mamba2_130m.json).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer path/ID (e.g. fla-hub/transformer-1.3B-100B).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for HF files.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save model weights as safetensors instead of pytorch_model.bin.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading state_dict.",
    )
    args = parser.parse_args()

    convert_pt_to_hf(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
