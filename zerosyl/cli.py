from pathlib import Path
from typing import List, Literal, Optional

import torch
import typer
from rich.progress import track
from torchcodec.decoders import AudioDecoder

from zerosyl import ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete

app = typer.Typer(add_completion=False)


@app.command()
def encode(
    input_path: Path = typer.Argument(..., help="Input file or directory."),
    output_path: Path = typer.Argument(..., help="Output file or directory."),
    output: Literal["continuous", "discrete", "collapsed"] = typer.Option(
        "collapsed",
        "--output",
        help='Choose output type: "continuous" or "discrete" or "collapsed".',
    ),
    extension: str = typer.Option(
        ".wav", "--extension", "-e", help="Audio extension to process."
    ),
):
    """
    Segment the waveform into syllable-like segments and save the output as a .pt file.

    -----------------------------------------------------------------------------

    - If input_path is a file: produce one output file.

    - If input_path is a directory: process all audio files inside.

    -----------------------------------------------------------------------------

    - If --output is "continuous", save the continuous representations.
    Each output file is a dictionary with keys: "starts", "ends" and "embeddings".
    "starts" and "ends" are long tensors containing frame indices.
    "embeddings" is a float32 tensor with one 1024-dimensional embedding per segment.

    - If --output is "discrete" save the segments with a K-means cluster ID (one of 10,000 types) associated with each segment. One of 10,000 values.

    - If --output is "collapsed" save the segments with an K-means cluster ID (one of 9,116 types) associated with each segment, where all the centroids that correspond to silences are mapped to a single id, SIL=9115.

    -----------------------------------------------------------------------------

    """

    input_path_list: List[Path] = None
    output_path_list: List[Path] = None

    if input_path.is_file():

        input_path_list = [input_path]

        if output_path is None:
            output_path_list = [None]
        elif output_path.is_dir():
            output_path_list = [output_path / input_path.with_suffix(".pt")]
        else:
            output_path_list = [output_path.with_suffix(".pt")]

    elif input_path.is_dir():

        typer.echo(f"Finding input files...")
        input_path_list = list(input_path.rglob(f"*{extension}"))

        if len(input_path_list) == 0:
            typer.echo(
                f"No {extension} files found. Check input_path or supply different extension with --extension",
                err=True,
            )
            raise typer.Exit(code=1)

        if output_path.is_file():
            typer.echo(
                "Output path cannot be a file if the input path is a directory",
                err=True,
            )
            raise typer.Exit(code=1)

        output_path_list = [
            output_path / p.relative_to(input_path).with_suffix(".pt")
            for p in input_path_list
        ]

    else:
        typer.echo("Input not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading model...")

    if output == "continuous":
        zerosyl_continuous = ZeroSylContinuous.from_remote().cuda()
    elif output == "discrete":
        zerosyl_discrete = ZeroSylDiscrete.from_remote().cuda()
    elif output == "collapsed":
        zerosyl_collapsed = ZeroSylCollapsed.from_remote().cuda()

    for input_path, output_path in track(
        zip(input_path_list, output_path_list),
        description="Encoding...",
        total=len(input_path_list),
        disable=len(input_path_list) <= 1,
    ):
        decoder = AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data.cuda()

        if output == "continuous":
            starts, ends, embeddings = zerosyl_continuous.encode(wav)
            output = {"starts": starts, "ends": ends, "embeddings": embeddings}
        elif output == "discrete":
            starts, ends, ids = zerosyl_discrete.encode(wav)
            output = torch.stack([starts, ends, ids], dim=1).cpu()
        elif output == "collapsed":
            starts, ends, ids = zerosyl_collapsed.encode(wav)
            output = torch.stack([starts, ends, ids], dim=1).cpu()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, output_path)

    typer.echo(f"Done!")


@app.command()
def decode(
    input_path: Path = typer.Argument(
        ..., help="Input file or directory with segments."
    ),
    output_path: Path = typer.Argument(
        ..., help="Output file or directory for waveforms."
    ),
    extension: str = typer.Option(
        ".wav", "--extension", "-e", help="Audio extension to use."
    ),
):
    """
    Synthesize audio from the given segments or ids

    -----------------------------------------------------------------------------

    - If input_path is a file: produce one output file.

    - If input_path is a directory: process all audio files inside.

    -----------------------------------------------------------------------------

    - If --output is "continuous", save the continuous representations.
    Each output file is a dictionary with keys: "starts", "ends" and "embeddings".
    "starts" and "ends" are long tensors containing frame indices.
    "embeddings" is a float32 tensor with one 1024-dimensional embedding per segment.

    - If --output is "discrete" save the segments with a K-means cluster ID (one of 10,000 types) associated with each segment. One of 10,000 values.

    - If --output is "collapsed" save the segments with an K-means cluster ID (one of 9,116 types) associated with each segment, where all the centroids that correspond to silences are mapped to a single id, SIL=9115.

    -----------------------------------------------------------------------------

    """

    input_path_list: List[Path] = None
    output_path_list: List[Path] = None

    if input_path.is_file():

        input_path_list = [input_path]

        if output_path is None:
            output_path_list = [None]
        elif output_path.is_dir():
            output_path_list = [output_path / input_path.with_suffix(".pt")]
        else:
            output_path_list = [output_path.with_suffix(".pt")]

    elif input_path.is_dir():

        typer.echo(f"Finding input files...")
        input_path_list = list(input_path.rglob(f"*{extension}"))

        if len(input_path_list) == 0:
            typer.echo(
                f"No {extension} files found. Check input_path or supply different extension with --extension",
                err=True,
            )
            raise typer.Exit(code=1)

        if output_path.is_file():
            typer.echo(
                "Output path cannot be a file if the input path is a directory",
                err=True,
            )
            raise typer.Exit(code=1)

        output_path_list = [
            output_path / p.relative_to(input_path).with_suffix(".pt")
            for p in input_path_list
        ]

    else:
        typer.echo("Input not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading model...")

    if output == "continuous":
        zerosyl_continuous = ZeroSylContinuous.from_remote().cuda()
    elif output == "discrete":
        zerosyl_discrete = ZeroSylDiscrete.from_remote().cuda()
    elif output == "collapsed":
        zerosyl_collapsed = ZeroSylCollapsed.from_remote().cuda()

    for input_path, output_path in track(
        zip(input_path_list, output_path_list),
        description="Encoding...",
        total=len(input_path_list),
        disable=len(input_path_list) <= 1,
    ):
        decoder = AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data.cuda()

        if output == "continuous":
            starts, ends, embeddings = zerosyl_continuous.encode(wav)
            output = {"starts": starts, "ends": ends, "embeddings": embeddings}
        elif output == "discrete":
            starts, ends, ids = zerosyl_discrete.encode(wav)
            output = torch.stack([starts, ends, ids], dim=1).cpu()
        elif output == "collapsed":
            starts, ends, ids = zerosyl_collapsed.encode(wav)
            output = torch.stack([starts, ends, ids], dim=1).cpu()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, output_path)

    typer.echo(f"Done!")
