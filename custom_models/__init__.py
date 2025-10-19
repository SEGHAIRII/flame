from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from .vanilla_attention.attention_config import AudioTransformerConfig
from .vanilla_attention.attention_model import AudioTransformerModel

from .mamba2_audio.config_mamba2 import AudioMamba2Config
from .mamba2_audio.model_mamba2 import AudioMamba2Model
from .sba.config_sba import SBAConfig
from .sba.modeling_sba import SBAForCausalLM, SBAModel

AutoConfig.register("audio_mamba2", AudioMamba2Config)
AutoModel.register(AudioMamba2Config, AudioMamba2Model)
AutoConfig.register('sba', SBAConfig)
AutoModel.register(SBAConfig, SBAModel)
AutoModelForCausalLM.register(SBAConfig, SBAForCausalLM)
AutoConfig.register("audio_attention", AudioTransformerConfig)
AutoModel.register(AudioTransformerConfig, AudioTransformerModel)

__all__ = ["AudioMamba2Config", "AudioMamba2Model", 'SBAConfig', 'SBAForCausalLM', 'SBAModel', 'AudioTransformerConfig', 'AudioTransformerModel']