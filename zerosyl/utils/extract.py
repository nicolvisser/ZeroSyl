from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from zerosyl.encoder import ZeroSylDiscrete


def extract_segments(
    checkpoint_path: str = "checkpoints/WavLM-Large.pt",
    centroids_path: str = "checkpoints/km10000-centroids-v040.pt",
    waveform_dir: str = "data/waveforms/LibriSpeech",
    waveform_pattern: str = "**/*.flac",
    output_dir: str = "temp/segments/LibriSpeech",
):
    assert Path(checkpoint_path).exists(), checkpoint_path
    waveform_dir = Path(waveform_dir)
    assert Path(waveform_dir).exists(), waveform_dir

    waveform_paths = list(Path(waveform_dir).glob(waveform_pattern))
    assert len(waveform_paths) > 0

    encoder = ZeroSylDiscrete.from_pretrained_checkpoint(
        checkpoint_path, centroids_path
    ).cuda()

    for waveform_path in tqdm(waveform_paths):
        rel_path = waveform_path.relative_to(Path(waveform_dir)).with_suffix(".pt")
        out_path = output_dir / rel_path
        if out_path.exists():
            continue

        decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data.cuda()

        starts, ends, ids = encoder.encode(wav)

        segments = torch.stack([starts, ends, ids], dim=1).cpu()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(segments, out_path)
