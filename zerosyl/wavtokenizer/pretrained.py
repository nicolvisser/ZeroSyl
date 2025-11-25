dependencies = ["torch", "torchaudio"]

URLS = {
    "small_600_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_600_24k_4096_d44c40fb.ckpt",
    "small_320_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_320_24k_4096_721a204f.ckpt",
}


import torch
import torchaudio
from typing import Callable, Tuple

from .configs import ARGS_SMALL_320_24K_4096, ARGS_SMALL_600_24K_4096
from .decoder.pretrained import WavTokenizer, WavTokenizerArgs

@torch.inference_mode()
def encode(model: torch.nn.Module, wav: torch.Tensor, sr: int) -> torch.Tensor:
    device = next(model.parameters()).device
    wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.to(device)
    bandwidth_id = torch.tensor([0], device=device)
    _, codes = model.encode(wav, bandwidth_id=bandwidth_id)
    return codes

@torch.inference_mode()
def decode(model: torch.nn.Module, codes: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    codes = codes.to(device)
    bandwidth_id = torch.tensor([0], device=device)
    features = model.codes_to_features(codes)
    audio_out = model.decode(features, bandwidth_id=bandwidth_id)
    sr = 24000
    return audio_out, sr

def _load(
    args: WavTokenizerArgs,
    url: str,
    progress: bool = True,
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 600x downsample (40 Hz), 4096 codebook entries."""

    model = WavTokenizer(args=args)
    state_dict_raw = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=progress, weights_only=True
    )["state_dict"]
    state_dict = dict()
    for k, v in state_dict_raw.items():
        if (
            k.startswith("backbone.")
            or k.startswith("head.")
            or k.startswith("feature_extractor.")
        ):
            state_dict[k] = v
    model.load_state_dict(state_dict)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"WavTokenizer loaded with {params:,} parameters")

    return model


def load_wavtokenizer_small_600_24k_4096(
    progress: bool = True
) -> WavTokenizer:
    model = _load(
        args=ARGS_SMALL_600_24K_4096,
        url=URLS["small_600_24k_4096"],
        progress=progress,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded WavTokenizer small_600_24k_4096 with {num_params:,} parameters")
    return model


def load_small_320_24k_4096(
    progress: bool = True
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 320x downsample (75 Hz), 4096 codebook entries."""
    model, encode, decode = _load(
        args=ARGS_SMALL_320_24K_4096,
        url=URLS["small_320_24k_4096"],
        progress=progress,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded WavTokenizer small_320_24k_4096 with {num_params:,} parameters")
    return model, encode, decode