import torch

assert torch.cuda.is_available(), (
    "ZeroSyl requires a CUDA-enabled GPU. "
    "Please install a CUDA build of PyTorch and ensure that a compatible GPU is available."
)

from neucodec import DistillNeuCodec, NeuCodec

from .acoustic import AcousticModel, AcousticModelConfig
from .encoder import ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete
from .ulm import ULM, ULMConfig
from .wavlm.WavLM import WavLM, WavLMConfig
