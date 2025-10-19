from typing import Optional, Union, List
from flame.models.base_audio_models.base_audio_config import AudioDenoiserConfig


class AudioMamba2Config(AudioDenoiserConfig):
    """
    Configuration class for AudioMamba2 model.
    
    This config extends AudioDenoiserConfig with Mamba2-specific parameters
    that match the FLA library's Mamba2Block implementation.
    """
    
    model_type = "audio_mamba2"
    
    def __init__(
        self,
        # Model Architecture
        num_hidden_layers: int = 6,
        hidden_size: int = 256,
        
        # Mamba2 layer parameters
        num_heads: int = 8,
        head_dim: int = 32,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        
        # Activation and normalization
        hidden_act: str = "silu",
        rms_norm: bool = True,
        norm_eps: float = 1e-5,
        residual_in_fp32: bool = True,
        
        # Time step parameters
        time_step_rank: Union[int, str] = "auto",
        time_step_limit: Optional[List[float]] = None,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        
        # Bias settings
        use_bias: bool = False,
        use_conv_bias: bool = True,
        
        # Audio-specific parameters
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
        
        # Training parameters
        loss_type: str = "mse",
        initializer_range: float = 0.02,
        
        # Other parameters
        rescale_prenorm_residual: bool = False,
        fuse_norm: bool = False,
        fuse_cross_entropy: bool = False,
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
            use_mel=True,  # Enable mel spectrogram mode
            use_magnitude=True,  # Use magnitude spectrum
            use_phase=False,  # Don't use phase
            use_complex=False,  # Don't use complex representation
            hidden_size=hidden_size,  # Pass to parent
            num_layers=num_hidden_layers,  # Pass to parent
            fuse_norm=fuse_norm,
            **kwargs
        )
        
        # Model architecture - OVERRIDE after parent init
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        
        # Mamba2 layer parameters
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.expand = expand
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        
        # Verify dimensions match
        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"num_heads ({num_heads}) * head_dim ({head_dim}) must equal "
            f"hidden_size ({hidden_size})"
        )
        
        # Activation and normalization
        self.hidden_act = hidden_act
        self.rms_norm = rms_norm
        self.norm_eps = norm_eps
        self.residual_in_fp32 = residual_in_fp32
        
        # Time step parameters
        self.time_step_rank = time_step_rank
        self.time_step_limit = time_step_limit if time_step_limit is not None else [0.001, 100.0]
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        
        # Bias settings
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        
        # Other parameters
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_cache = use_cache
        self.hidden_act = hidden_act
        self.rms_norm = rms_norm
        self.norm_eps = norm_eps
        self.norm_epsilon = norm_eps  # Add this - used by model
        self.residual_in_fp32 = residual_in_fp32