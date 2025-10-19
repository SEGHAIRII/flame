import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from flame.models.base_audio_models.base_audio_model import AudioDenoiserBase, BackboneWrapper
from flame.models.base_audio_models.AudioDenoiserOutput import AudioDenoiserOutput
from .attention_config import AudioTransformerConfig


class AudioTransformerModel(AudioDenoiserBase):
    """Vanilla Transformer-based audio denoising model"""
    config_class = AudioTransformerConfig
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__(config)
        
        # Positional encoding
        if config.use_positional_encoding:
            if config.positional_encoding_type == "learnable":
                self.positional_encoding = nn.Parameter(
                    torch.zeros(1, config.max_position_embeddings, config.hidden_size)
                )
            elif config.positional_encoding_type == "sinusoidal":
                self.register_buffer(
                    "positional_encoding",
                    self._create_sinusoidal_positions(
                        config.max_position_embeddings, 
                        config.hidden_size
                    )
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {config.positional_encoding_type}")
        else:
            self.positional_encoding = None
        
        # Build backbone
        self.layers = self.build_backbone()
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def build_backbone(self) -> nn.ModuleList:
        """Build Transformer encoder layers"""
        layers = nn.ModuleList([
            TransformerEncoderLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        ])
        
        return TransformerBackbone(layers)
    
    def _create_sinusoidal_positions(self, num_positions: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(1, num_positions, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> AudioDenoiserOutput:
        
        
        batch_size, seq_len, feature_dim = input_ids.shape
        
        assert feature_dim == self.config.audio_feature_dim, (
            f"Expected feature_dim={self.config.audio_feature_dim}, got {feature_dim}"
        )
        
        # Project to hidden dimension
        hidden_states = self.input_projection(input_ids)
        
        # Add positional encoding
        if self.positional_encoding is not None:
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_position_embeddings "
                    f"{self.config.max_position_embeddings}"
                )
            hidden_states = hidden_states + self.positional_encoding[:, :seq_len, :]
        
        # Apply transformer layers
        hidden_states = self.layers(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project back to feature space
        denoised_features = self.output_projection(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = self.criterion(denoised_features, labels)
        
        # Return output
        if not return_dict:
            output = (denoised_features,)
            if loss is not None:
                output = (loss,) + output
            return output + (hidden_states,)
        
        return AudioDenoiserOutput(
            loss=loss,
            denoised_features=denoised_features,
            hidden_states=hidden_states,
            attentions=None,
        )


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with self-attention and FFN"""
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        
        # Self-attention
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Self-attention with residual connection and layer norm (pre-norm)
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN with residual connection and layer norm (pre-norm)
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_size = config.hidden_size
        
        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Activation function
        if config.hidden_act == "gelu":
            self.activation = F.gelu
        elif config.hidden_act == "relu":
            self.activation = F.relu
        elif config.hidden_act == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {config.hidden_act}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TransformerBackbone(BackboneWrapper):
    """Transformer backbone wrapper"""
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__(layers)
        self.layers = layers
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states