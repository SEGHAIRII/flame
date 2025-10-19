from typing import Optional
from flame.models.base_audio_models.base_audio_config import AudioDenoiserConfig


class AudioTransformerConfig(AudioDenoiserConfig):
    """
    Configuration class for Transformer-based audio denoising model.
    
    Uses standard multi-head self-attention with feed-forward layers.
    """
    
    model_type = "audio_attention"
    
    def __init__(
        self,
        # Model Architecture
        num_hidden_layers: int = 6,
        hidden_size: int = 256,
        
        # Attention parameters
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        
        # Feed-forward parameters
        intermediate_size: int = 1024,  # FFN hidden dimension
        hidden_dropout: float = 0.1,
        
        # Activation and normalization
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        
        # Position encoding
        max_position_embeddings: int = 4096,  # Max sequence length
        use_positional_encoding: bool = True,
        positional_encoding_type: str = "learnable",  # "learnable" or "sinusoidal"
        
        # Audio-specific parameters
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
        
        # Training parameters
        loss_type: str = "mse",
        initializer_range: float = 0.02,
        
        # Other parameters
        use_cache: bool = False,
        
        **kwargs
    ):
        # Initialize parent class FIRST with audio parameters
        super().__init__(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            loss_type=loss_type,
            hidden_size=hidden_size,
            num_layers=num_hidden_layers,
            **kwargs
        )
        
        # Model architecture
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        
        # Attention parameters
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        # Verify attention heads divide hidden size
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
        self.attention_head_dim = hidden_size // num_attention_heads
        
        # Feed-forward parameters
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        
        # Activation and normalization
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        
        # Position encoding
        self.max_position_embeddings = max_position_embeddings
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        
        # Other parameters
        self.initializer_range = initializer_range
        self.use_cache = use_cache