from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class AudioDenoiserConfig(PretrainedConfig):
    model_type = "audio_denoiser"
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,  # Defaults to n_fft if None
        window: str = "hann",  # Window type: "hann", "hamming", "blackman"
        center: bool = True,  # Center the window
        normalized: bool = False,  # Normalize STFT
        onesided: bool = True,  # Return only positive frequencies
        
        # Mel spectrogram configuration
        use_mel: bool = False,  # Use mel spectrogram instead of raw STFT
        n_mels: int = 80,  # Number of mel filterbanks
        
        n_freq_bins: Optional[int] = None,  # Auto-computed: n_fft // 2 + 1 if onesided
        use_magnitude: bool = True,  # Use magnitude spectrum
        use_phase: bool = False,  # Use phase spectrum
        use_complex: bool = False,  # Use complex-valued features (real + imag)
        
        hidden_size: int = 512,
        num_layers: int = 6,
        
        loss_type: str = "mse",  # "mse", "l1", "smooth_l1", "huber", "spectral"
        
        fuse_norm: bool = False,
        fuse_linear_cross_entropy: bool = False,  
        vocab_size: int = 1,  # Dummy for compatibility
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # STFT configuration
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        
        # Mel spectrogram configuration
        self.use_mel = use_mel
        self.n_mels = n_mels
        
        # Compute frequency bins
        if n_freq_bins is None:
            self.n_freq_bins = (n_fft // 2 + 1) if onesided else n_fft
        else:
            self.n_freq_bins = n_freq_bins
        
        # Feature representation
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase
        self.use_complex = use_complex
        
        # Compute actual audio feature dimension based on representation
        self.audio_feature_dim = self._compute_feature_dim()
        
        # Model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Loss configuration
        self.loss_type = loss_type
        
        # Training compatibility
        self.fuse_norm = fuse_norm
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.vocab_size = vocab_size
    
    def _compute_feature_dim(self) -> int:
        """Compute the input feature dimension based on STFT representation"""
        feature_dim = 0
        
        # Determine the base number of frequency bins
        if self.use_mel:
            # When using mel spectrogram, use n_mels instead of n_freq_bins
            base_bins = self.n_mels
        else:
            # Use raw STFT frequency bins
            base_bins = self.n_freq_bins
        
        if self.use_complex:
            # Complex representation: real + imaginary = 2 * base_bins
            feature_dim = 2 * base_bins
        else:
            # Separate magnitude and phase
            if self.use_magnitude:
                feature_dim += base_bins
            if self.use_phase:
                feature_dim += base_bins
        
        if feature_dim == 0:
            raise ValueError(
                "Must enable at least one of: use_magnitude, use_phase, or use_complex"
            )
        
        return feature_dim
    
    def get_stft_params(self) -> Dict[str, Any]:
        """Return dictionary of STFT parameters for torch.stft"""
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "return_complex": True,
        }