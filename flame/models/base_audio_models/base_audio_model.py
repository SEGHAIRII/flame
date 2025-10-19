import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Optional
from abc import ABC, abstractmethod

from flame.models.base_audio_models.base_audio_config import AudioDenoiserConfig
from .AudioDenoiserOutput import AudioDenoiserOutput


class AudioDenoiserBase(PreTrainedModel):
    """Base class for all audio denoising models"""
    config_class = AudioDenoiserConfig
    base_model_prefix = ""
    
    def __init__(self, config: AudioDenoiserConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection: audio features -> hidden dimension
        self.input_projection = nn.Linear(
            config.audio_feature_dim, 
            config.hidden_size
        )
        
        # Output projection: hidden dimension -> audio features
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.audio_feature_dim
        )
        
        # Loss function
        self.criterion = self._build_loss_fn(config.loss_type)
        
        # Backbone (implemented by subclasses)
        self.backbone = None
    
    def _build_loss_fn(self, loss_type: str) -> nn.Module:
        """Build regression loss function"""
        loss_functions = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(),
            "huber": nn.HuberLoss(),
            "spectral": SpectralLoss(),
        }
        
        if loss_type not in loss_functions:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_functions[loss_type]
    
    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Subclasses must implement this"""
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T, F] - noisy audio features
        labels: Optional[torch.Tensor] = None,  # [B, T, F] - clean audio
        position_ids: Optional[torch.Tensor] = None,  # For compatibility
        cu_seqlens: Optional[torch.Tensor] = None,  # For compatibility
        attention_mask: Optional[torch.Tensor] = None,  # For compatibility
        return_dict: bool = True,  # Standard HF argument
        **kwargs
    ) -> AudioDenoiserOutput:
        """
        Forward pass for audio denoising.
        
        Args:
            input_ids: Noisy audio features [batch, time_steps, feature_dim]
            labels: Clean audio features [batch, time_steps, feature_dim]
            position_ids: Not used (kept for train.py compatibility)
            cu_seqlens: Not used (kept for train.py compatibility)
            attention_mask: Optional attention mask
            return_dict: Whether to return AudioDenoiserOutput (True) or tuple
        
        Returns:
            AudioDenoiserOutput with loss and denoised features
        """
        batch_size, seq_len, feature_dim = input_ids.shape
        
        # Validate input
        assert feature_dim == self.config.audio_feature_dim, (
            f"Expected feature_dim={self.config.audio_feature_dim}, got {feature_dim}"
        )
        
        # Project to hidden dimension
        hidden_states = self.input_projection(input_ids)  # [B, T, H]
        
        # Apply backbone
        if self.backbone is not None:
            hidden_states = self.backbone(hidden_states)  # [B, T, H]
        
        # Project back to feature space
        denoised_features = self.output_projection(hidden_states)  # [B, T, F]
        
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
    
    def post_init(self):
        """Weight initialization (called by train.py)"""
        if not getattr(self, "_is_hf_initialized", False):
            self.apply(self._init_weights)
            self._is_hf_initialized = True
    
    def _init_weights(self, module):
        """Default weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def denoise(
        self, 
        noisy_stft: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Args:
            noisy_stft: Complex spectrogram [B, F, T]
            return_features: If True, return features instead of STFT
        
        Returns:
            Denoised complex spectrogram [B, F, T] or features [B, T, F]
        """
        # Convert to features
        noisy_features = self.config.convert_complex_to_features(noisy_stft)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(input_ids=noisy_features, labels=None)
        
        denoised_features = output.denoised_features  # Changed from .logits
        
        if return_features:
            return denoised_features
        
        # Convert back to complex STFT
        if self.config.use_magnitude and not self.config.use_phase and not self.config.use_real_imag:
            original_phase = torch.angle(noisy_stft)
            denoised_stft = self.config.convert_features_to_complex(
                denoised_features, 
                original_phase=original_phase
            )
        else:
            denoised_stft = self.config.convert_features_to_complex(denoised_features)
        
        return denoised_stft
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding and hasattr(self, 'position_embeddings'):
            n_params -= self.position_embeddings.weight.numel()
        
        return n_params


class SpectralLoss(nn.Module):
    """Custom spectral loss"""
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


class BackboneWrapper(nn.Module):
    """Wrapper for backbone modules to provide consistent interface"""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through backbone"""
        return self.backbone(hidden_states)