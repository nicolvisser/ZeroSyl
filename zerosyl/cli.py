from pathlib import Path
from typing import Literal

import torch
import typer
from rich.progress import track
from torchcodec.decoders import AudioDecoder

from zerosyl import AcousticModel, ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete

app = typer.Typer(add_completion=False)

eval_app = typer.Typer(help="Evaluation utilities")
app.add_typer(eval_app, name="eval")


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
    Segment waveform(s) into syllabic segments and save the output as .pt file(s).

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

    input_path_list: list[Path] = None
    output_path_list: list[Path] = None

    if input_path.is_file():
        input_path_list = [input_path]
        output_path_list = [output_path]

    elif input_path.is_dir():

        typer.echo(f"Finding input files...")
        input_path_list = sorted(input_path.rglob(f"*{extension}"))

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

        rel_path_list = [p.relative_to(input_path) for p in input_path_list]
        output_path_list = [output_path / p.with_suffix(".pt") for p in rel_path_list]

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
            output_data = torch.stack([starts, ends, ids], dim=1).cpu()
        elif output == "collapsed":
            starts, ends, ids = zerosyl_collapsed.encode(wav)
            output_data = torch.stack([starts, ends, ids], dim=1).cpu()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_data, output_path)

    typer.echo(f"Done!")


@eval_app.command()
def boundaries(
    segments_dir: Path = typer.Argument(
        ..., help="Directory containing segment files."
    ),
    textgrid_dir: Path = typer.Argument(
        ..., help="Directory containing TextGrid files."
    ),
    segments_pattern: str = typer.Option(
        "dev*/**/*.pt", help="Glob pattern for segment files."
    ),
    textgrid_pattern: str = typer.Option(
        "dev*/**/*.TextGrid", help="Glob pattern for TextGrid files."
    ),
    frame_rate: float = typer.Option(50.0, help="Frame rate (Hz)."),
    constant_shift: float = typer.Option(
        0.0, help="Constant time shift applied to boundaries (in seconds)."
    ),
    tolerance: float = typer.Option(
        0.05, help="Tolerance for matching boundaries (in seconds)."
    ),
):
    """
    Compute the clustering metrics of a segmentation.
    """
    from zerosyl.eval.boundaries import evaluate_boundary_metrics

    if constant_shift == 0.0:
        typer.echo(
            "INFO: You started the boundary evaluation without supplying a constant shift."
        )
        typer.echo("Just remember, in our paper we tuned the constant shift.")

    precision, recall, f1, os, rvalue, token_precision, token_recall, token_f1 = (
        evaluate_boundary_metrics(
            segments_dir=segments_dir,
            textgrid_dir=textgrid_dir,
            segments_pattern=segments_pattern,
            textgrid_pattern=textgrid_pattern,
            frame_rate=frame_rate,
            constant_shift=constant_shift,
            tolerance=tolerance,
        )
    )

    typer.echo(f"Precision       {precision*100:8.2f}")
    typer.echo(f"Recall          {recall*100:8.2f}")
    typer.echo(f"F1              {f1*100:8.2f}")
    typer.echo(f"OS              {os*100:8.2f}")
    typer.echo(f"R-value         {rvalue*100:8.2f}")
    typer.echo()
    typer.echo(f"Token Precision {token_precision*100:8.2f}")
    typer.echo(f"Token Recall    {token_recall*100:8.2f}")
    typer.echo(f"Token F1        {token_f1*100:8.2f}")
    typer.echo()


@eval_app.command()
def bitrate(
    segments_dir: Path = typer.Argument(
        ..., help="Directory containing segment files."
    ),
    textgrid_dir: Path = typer.Argument(..., help="Directory with TextGrid files."),
    segments_pattern: str = typer.Option(
        "dev*/**/*.pt", help="Pattern for segment files."
    ),
    textgrid_pattern: str = typer.Option(
        "dev*/**/*.TextGrid", help="Pattern for TextGrid files."
    ),
    vocab_size: int = typer.Option(
        None, help="Optional vocab size. Otherwise, will infer the vocab size."
    ),
):
    """
    Compute the bitrate and frequecy of a segmentation.
    """
    from zerosyl.eval.bitrate import evaluate_bitrate_and_freq

    bitrate_value, freq_value = evaluate_bitrate_and_freq(
        segments_dir=segments_dir,
        textgrid_dir=textgrid_dir,
        segments_pattern=segments_pattern,
        textgrid_pattern=textgrid_pattern,
        vocab_size=vocab_size,
    )

    typer.echo(f"Bitrate:   {bitrate_value:10.2f} bits/s")
    typer.echo(f"Frequency: {freq_value:10.2f} Hz")


@eval_app.command()
def clustering(
    segments_dir: Path = typer.Argument(
        ..., help="Directory containing segment files."
    ),
    textgrid_dir: Path = typer.Argument(..., help="Directory with TextGrid files."),
    segments_pattern: str = typer.Option(
        "dev*/**/*.pt", help="Pattern for segment files."
    ),
    textgrid_pattern: str = typer.Option(
        "dev*/**/*.TextGrid", help="Pattern for TextGrid files."
    ),
):
    """
    Compute the clustering metrics of a segmentation.
    """
    from zerosyl.eval.clustering import evaluate_clustering_metrics

    per_cluster_purity, per_syllable_purity, snmi = evaluate_clustering_metrics(
        segments_dir=segments_dir,
        textgrid_dir=textgrid_dir,
        segments_pattern=segments_pattern,
        textgrid_pattern=textgrid_pattern,
    )

    typer.echo(f"Per-cluster purity:              {per_cluster_purity:10.4f}")
    typer.echo(f"Per-syllable purity:             {per_syllable_purity:10.4f}")
    typer.echo(f"Syllable-normalized mutual info: {snmi:10.4f}")


@eval_app.command()
def loglikelihood(
    segments_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing .pt segment files.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Where to write the log-likelihood text file.",
    ),
    checkpoint_path: str = typer.Option(
        None,
        help="Optional path or URL to the model checkpoint. Leave empty to download the default model remotely.",
    ),
    batch_size: int = typer.Option(
        64,
        min=1,
        help="Batch size for evaluation.",
    ),
    num_workers: int = typer.Option(
        8,
        min=0,
        help="Number of dataloader worker processes.",
    ),
    segments_pattern: str = typer.Option(
        "*.pt", help="Glob pattern to match segment files."
    ),
):
    """
    Compute log-likelihoods for a directory of unit segment files using a ULM checkpoint.

    - Output is a text file where each line contains: <segment-file-stem> <log-likelihood>.

    - Supports optional checkpoint: if none is provided, the model is downloaded remotely.
    """

    from zerosyl.eval.loglikelihood import compute_loglikelihoods

    if output_path.suffix.lower() != ".txt":
        raise typer.BadParameter("output_path must have a .txt extension")

    compute_loglikelihoods(
        segments_dir=segments_dir,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        batch_size=batch_size,
        num_workers=num_workers,
        segments_pattern=segments_pattern,
    )
