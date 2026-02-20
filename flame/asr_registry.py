
import os
from typing import Dict, Callable
import torch
import torch.nn as nn

from flame.config_manager import JobConfig
from torchtitan.tools.logging import logger as titan_logger


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable] = {}


def register_asr_model(name: str):
    """
    Decorator that registers a build function under `name`.

    Usage:
        @register_asr_model("hgrn_asr")
        def build_hgrn(job_config, metadata):
            ...
            return model
    """
    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            titan_logger.warning(f"[asr_registry] Overwriting existing entry '{name}'")
        _REGISTRY[name] = fn
        titan_logger.info(f"[asr_registry] Registered model: '{name}'")
        return fn
    return decorator


def build_model(name: str, job_config: JobConfig, metadata: dict) -> nn.Module:
    """
    Called by the training script. Looks up the registered build function and
    calls it with (job_config, metadata).
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {available}. "
            f"Did you forget to import the file that calls @register_asr_model?"
        )
    titan_logger.info(f"[asr_registry] Building model '{name}'")
    return _REGISTRY[name](job_config, metadata)


def list_models():
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helper: read a hyperparameter from job_config.model, with a fallback default.
# Keeps build functions tidy.
# ---------------------------------------------------------------------------

def _cfg(job_config: JobConfig, key: str, default):
    return getattr(job_config.model, key, default)


# ===========================================================================
# HGRN ASR  (built-in CTC head)
# ===========================================================================
@register_asr_model("hgrn_asr")
def _build_hgrn_asr(job_config: JobConfig, metadata: dict) -> nn.Module:
    from flame.models.hgrn_asr import HGRNASRConfig, HGRNASRForCTC 

    config_path = _cfg(job_config, "config", None)

    if config_path and os.path.exists(config_path):
        config = HGRNASRConfig.from_pretrained(config_path)
    else:
        config = HGRNASRConfig(
            input_size=metadata["input_dim"],
            vocab_size=metadata["n_classes"],
            max_sequence_length=max(metadata["seq_len"], 1),
            blank_id=0,
            hidden_size=_cfg(job_config, "hidden_size", 512),
            num_hidden_layers=_cfg(job_config, "num_layers", 6),
            expand_ratio=_cfg(job_config, "expand_ratio", 1),
            attn_mode=_cfg(job_config, "attn_mode", "chunk"),
            use_short_conv=False,
            conv_size=4,
            use_lower_bound=True,
            hidden_ratio=4,
            hidden_act="swish",
            elementwise_affine=True,
            norm_eps=1e-4,
            fuse_norm=True,
            fuse_swiglu=True,
        )

    # Option 1: Create directly on the target device (simplest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGRNASRForCTC(config).to(device)
    
    # Option 2: If you must use meta tensors for some reason, materialize them
    # with torch.device("meta"):
    #     model = HGRNASRForCTC(config)
    # # Materialize from meta tensors
    # model = model.to_empty(device="cuda")  # or .to("cuda")
    
    return model


@register_asr_model("sparse_mamba2_asr")
def _build_sparse_mamba2_asr(job_config: JobConfig, metadata: dict) -> nn.Module:
    from sparse_mamba.custom_models.audio_models.sparse_mamba2_ctc import SparseMamba2ASRConfig, SparseMamba2ASRForCTC

    config_path = _cfg(job_config, "config", None)

    if config_path and os.path.exists(config_path):
        config = SparseMamba2ASRConfig.from_pretrained(config_path)
    else:
        config = SparseMamba2ASRConfig(
            # ASR-specific
            input_size=metadata["input_dim"],
            vocab_size=metadata["n_classes"],
            max_sequence_length=max(metadata.get("seq_len", 0), 1),
            blank_id=0,

            # Main tunable hyperparameters (pulled from job config)
            hidden_size=_cfg(job_config, "hidden_size", 768),
            num_hidden_layers=_cfg(job_config, "num_layers", 12),
            state_size=_cfg(job_config, "state_size", 64),
            expand=_cfg(job_config, "expand", 2),
            conv_kernel=_cfg(job_config, "conv_kernel", 4),
            n_groups=_cfg(job_config, "n_groups", 1),
            chunk_size=_cfg(job_config, "chunk_size", 256),

            # Sparsity (default OFF for ASR speed)
            return_activation_sparsity=_cfg(job_config, "return_activation_sparsity", False),

            # Other reasonable defaults for ASR (you can override via job_config)
            head_dim=_cfg(job_config, "head_dim", 64),
            time_step_rank="auto",
            rms_norm=True,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            norm_eps=1e-5,
            residual_in_fp32=True,
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseMamba2ASRForCTC(config).to(device)
# No meta device, no to_empty, no post_init dance

    # Option 1: Create directly on the target device (simplest & recommended)
    # with torch.device("meta"):
    #     model = SparseMamba2ASRForCTC(config)
    
    # Option 2: If you need meta-device initialization first (e.g. for huge models)
    # with torch.device("meta"):
    #     model = SparseMamba2ASRForCTC(config)
    # model = model.to_empty(device="cuda")

    return model