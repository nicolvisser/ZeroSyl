from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from ..model import ZeroSylBase, ZeroSylDiscrete


def dump_embeddings_only(
    checkpoint_path: str,
    waveform_dir: str,
    waveform_pattern: str,
    output_dir: str,
):
    checkpoint_path = Path(checkpoint_path)
    waveform_dir = Path(waveform_dir)
    output_dir = Path(output_dir)

    assert checkpoint_path.exists()
    assert waveform_dir.exists()

    waveform_paths = list(waveform_dir.glob(waveform_pattern))
    assert len(waveform_paths) > 0

    model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()

    for waveform_path in tqdm(waveform_paths):
        decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()
        embeddings, _, _ = model.segment(audio.data.cuda())
        embeddings = embeddings.cpu()
        rel_path = waveform_path.relative_to(waveform_dir)
        out_path = output_dir / rel_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, out_path)


def dump_embeddings_and_lengths(
    checkpoint_path: str,
    waveform_dir: str,
    waveform_pattern: str,
    output_dir: str,
):
    checkpoint_path = Path(checkpoint_path)
    waveform_dir = Path(waveform_dir)
    output_dir = Path(output_dir)

    assert checkpoint_path.exists()
    assert waveform_dir.exists()

    waveform_paths = list(waveform_dir.glob(waveform_pattern))
    assert len(waveform_paths) > 0

    model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()

    for waveform_path in tqdm(waveform_paths):
        decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()
        embeddings, starts, ends = model.segment(audio.data.cuda())
        lengths = ends - starts
        data = {"embeddings": embeddings.cpu(), "lengths": lengths.cpu()}
        rel_path = waveform_path.relative_to(waveform_dir)
        out_path = output_dir / rel_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)


def dump_tokens_only(
    checkpoint_path: str,
    centroids_path: str,
    waveform_dir: str,
    waveform_pattern: str,
    output_dir: str,
):
    checkpoint_path = Path(checkpoint_path)
    centroids_path = Path(centroids_path)
    waveform_dir = Path(waveform_dir)
    output_dir = Path(output_dir)

    assert checkpoint_path.exists()
    assert checkpoint_path.exists()
    assert waveform_dir.exists()

    waveform_paths = list(waveform_dir.glob(waveform_pattern))
    assert len(waveform_paths) > 0

    model = ZeroSylDiscrete.from_pretrained_checkpoint(
        checkpoint_path, centroids_path
    ).cuda()

    for waveform_path in tqdm(waveform_paths):
        decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()
        tokens, _, _ = model.tokenize(audio.data.cuda())
        tokens = tokens.cpu()
        rel_path = waveform_path.relative_to(waveform_dir)
        out_path = output_dir / rel_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, out_path)


def dump_tokens_and_lengths(
    checkpoint_path: str,
    centroids_path: str,
    waveform_dir: str,
    waveform_pattern: str,
    output_dir: str,
):
    checkpoint_path = Path(checkpoint_path)
    centroids_path = Path(centroids_path)
    waveform_dir = Path(waveform_dir)
    output_dir = Path(output_dir)

    assert checkpoint_path.exists()
    assert checkpoint_path.exists()
    assert waveform_dir.exists()

    waveform_paths = list(waveform_dir.glob(waveform_pattern))
    assert len(waveform_paths) > 0

    model = ZeroSylDiscrete.from_pretrained_checkpoint(
        checkpoint_path, centroids_path
    ).cuda()

    for waveform_path in tqdm(waveform_paths):
        decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()
        tokens, starts, ends = model.tokenize(audio.data.cuda())
        lengths = ends - starts
        data = {"tokens": tokens.cpu(), "lengths": lengths.cpu()}
        rel_path = waveform_path.relative_to(waveform_dir)
        out_path = output_dir / rel_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
