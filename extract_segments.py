from pathlib import Path

from zerosyl.encoder import ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete


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

    encoder = ZeroSylContinuous.from_pretrained_checkpoint(checkpoint_path)

    for waveform_path in waveform_paths:
        rel_path = waveform_path.relative_to(Path(waveform_dir)).with_suffix
