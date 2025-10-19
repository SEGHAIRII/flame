from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class AudioDenoiserOutput:
    """
    Output type for audio denoising models.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Denoising loss (MSE, L1, etc.) between predicted and target audio.
        denoised_features (`torch.FloatTensor` of shape `(batch_size, time_steps, feature_dim)`):
            Denoised audio features (magnitude, real/imag, or magnitude+phase).
        hidden_states (`torch.FloatTensor` of shape `(batch_size, time_steps, hidden_size)`, *optional*):
            Hidden states from the backbone for analysis/visualization.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights (if applicable). None for SSM-based models.
    """
    loss: Optional[torch.FloatTensor] = None
    denoised_features: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility"""
        if key == "loss":
            return self.loss
        elif key == "logits":  # Alias for compatibility with train.py
            return self.denoised_features
        elif key == "hidden_states":
            return self.hidden_states
        elif key == "attentions":
            return self.attentions
        else:
            raise KeyError(f"'{key}' not found in AudioDenoiserOutput")
    
    def to_tuple(self):
        """Convert to tuple for compatibility"""
        return tuple(
            v for v in [
                self.loss,
                self.denoised_features,
                self.hidden_states,
                self.attentions,
            ] if v is not None
        )


@dataclass
class AudioDenoiserOutputWithPast(AudioDenoiserOutput):
    """
    Extended output with past key values (for future caching support).
    Currently not used but kept for API consistency.
    """
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None