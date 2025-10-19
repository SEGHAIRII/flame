import torch
import torch.nn as nn
from typing import Optional

from flame.models.base_audio_models.base_audio_config import AudioDenoiserConfig
from flame.models.base_audio_models.base_audio_model import AudioDenoiserBase, BackboneWrapper
from flame.models.base_audio_models.AudioDenoiserOutput import AudioDenoiserOutput

from .config_mamba2 import AudioMamba2Config

# Import from fla library
from fla.models.mamba2.modeling_mamba2 import Mamba2Block
from fla.modules.layernorm import RMSNorm


class AudioMamba2Model(AudioDenoiserBase):
    """Mamba 2-based audio denoising model using fla library"""
    config_class = AudioMamba2Config
    
    def __init__(self, config: AudioMamba2Config):
        super().__init__(config)
        
        # Build backbone
        self.layers = self.build_backbone()
        
        # Final layer norm
        if config.rms_norm:
            self.norm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.norm_epsilon
            )
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
            
        self.apply(self._init_weights)
    
    def build_backbone(self) -> nn.ModuleList:
        """Build Mamba 2 backbone using fla.models.mamba2.Mamba2Block"""
        
        layers = nn.ModuleList()
        
        for layer_idx in range(self.config.num_hidden_layers):
            # ✅ Create Mamba2Block with all required parameters
            block = Mamba2Block(
                config=self.config,  # Pass the entire config
                layer_idx=layer_idx  # Required for layer-specific initialization
            )
            layers.append(block)
        
        return Mamba2Backbone(layers)
    
    
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight'):
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor = None,  # Can be None
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> AudioDenoiserOutput:
        
        
        batch_size, seq_len, feature_dim = input_ids.shape
        input_ids = input_ids.view(batch_size, seq_len, feature_dim)
        labels = labels.view(batch_size, seq_len, feature_dim) if labels is not None else None
        
        # assert feature_dim == self.config.audio_feature_dim, (
        #     f"Expected feature_dim={self.config.audio_feature_dim}, got {feature_dim}"
        # )
        
        # Project to hidden dimension
        hidden_states = self.input_projection(input_ids)
        
        # Apply backbone
        hidden_states = self.layers(
            hidden_states=hidden_states,
            position_ids=position_ids,
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
        
        # Return custom output
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


class Mamba2Backbone(BackboneWrapper):
    """Mamba 2 backbone using fla's Mamba2Block"""
    def __init__(self, layers: nn.ModuleList):
        nn.Module.__init__(self)
        self.layers = layers
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through all Mamba 2 blocks"""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states