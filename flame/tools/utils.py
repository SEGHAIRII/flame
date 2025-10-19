# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchtitan.tools.logging import logger


def get_nparams_and_flops(model: nn.Module, model_config, seq_len: int) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )
    
    # Handle different model types
    if hasattr(model_config, "num_heads"):
        num_heads = model_config.num_heads
    elif hasattr(model_config, "num_attention_heads"):
        num_heads = model_config.num_attention_heads
    else:
        num_heads = 1
        logger.warning("num_heads not found in model_config, defaulting to 1. ")

    # Get number of layers
    if hasattr(model_config, "num_hidden_layers"):
        num_layers = model_config.num_hidden_layers
    elif hasattr(model_config, "num_layers"):
        num_layers = model_config.num_layers
    else:
        num_layers = 1
        logger.warning("num_layers not found in model_config, defaulting to 1. ")

    l, h, q, t = (
        num_layers,
        num_heads,
        model_config.hidden_size // num_heads,
        seq_len,
    )
    
    # Check if this is a language model or audio model
    is_audio_model = hasattr(model_config, "audio_feature_dim")
    
    if is_audio_model:
        # For audio models, use a simplified FLOP calculation
        # Each linear layer: 2 * in_features * out_features operations per token
        # Each attention layer: 4 * hidden_size^2 (Q, K, V, O projections)
        # Each FFN layer: 2 * hidden_size * intermediate_size * 2
        
        hidden_size = model_config.hidden_size
        
        if hasattr(model_config, "intermediate_size"):
            intermediate_size = model_config.intermediate_size
        else:
            # Default FFN expansion factor of 4
            intermediate_size = hidden_size * 4
        
        # Approximate FLOPs per token:
        # - Attention: 4 operations (Q, K, V, O) each with hidden_size^2
        # - Attention computation: 2 * hidden_size * seq_len (scaled dot-product)
        # - FFN: 2 operations (up, down) with hidden_size * intermediate_size
        flops_per_layer = (
            4 * 2 * hidden_size * hidden_size +  # QKV + O projections
            4 * hidden_size * t +                 # Attention scores computation
            2 * 2 * hidden_size * intermediate_size  # FFN up and down
        )
        
        num_flops_per_token = l * flops_per_layer
        
        logger.info(
            f"Audio model FLOP calculation: {l} layers, "
            f"{hidden_size} hidden_size, {intermediate_size} intermediate_size, "
            f"{t} seq_len -> {num_flops_per_token} FLOPs/token"
        )
    else:
        # Original calculation for language models with embeddings
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    return nparams, num_flops_per_token