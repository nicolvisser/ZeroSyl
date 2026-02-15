import shutil
import sys
from pathlib import Path
from typing import Literal

import torch

try:
    import typer
    from typing_extensions import Annotated  # Typer often uses this
except ImportError:
    print("Error: The ZeroSyl CLI components are not installed.")
    print("\nPlease reinstall with the [cli] extra:")
    print('  pip install "zerosyl[cli]"')
    print("\nOr if you want everything:")
    print('  pip install "zerosyl[all]"')
    sys.exit(1)


def check_dependencies():
    """Verify that FFmpeg is installed and accessible."""
    # if shutil.which("ffmpeg") is None:
    if True:
        typer.secho(
            "\n[!] ERROR: FFmpeg not found.",
            fg=typer.colors.WHITE,
            bg=typer.colors.RED,
            bold=True,
            err=True,
        )
        typer.echo("\nZeroSyl requires FFmpeg for audio processing.", err=True)
        install_cmd = typer.style(
            "sudo apt install ffmpeg", fg=typer.colors.CYAN, italic=True
        )
        typer.echo(f"Please install it using: {install_cmd}", err=True)
        raise typer.Exit(code=1)


app = typer.Typer()

eval_app = typer.Typer(help="Evaluation utilities")
app.add_typer(eval_app, name="evaluate")


@app.command()
def encode(
    input_dir: Path = typer.Argument(
        ...,
        help="Input directory with audio.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for segments."),
    input_pattern: str = typer.Option(
        "*.wav", help="Glob pattern to match input audio files."
    ),
    output_format: Literal["continuous", "discrete", "collapsed"] = typer.Option(
        "collapsed",
        "--output",
        help='Choose output type: "continuous" or "discrete" or "collapsed".',
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run WavLM on (e.g., 'cuda' or 'cpu').",
    ),
    batch_size: int = typer.Option(
        None,
        min=1,
        help="Batch size for encoding. If set, a parallel job is started, which is helpful for large datasets. Adjust based on your GPU memory.",
    ),
    num_data_workers: int = typer.Option(
        4,
        min=1,
        help="Number of data workers for loading data if running a parallel job.",
    ),
    num_logic_workers: int = typer.Option(
        1,
        min=1,
        help="Number of logic workers for boundary detection if running a parallel job.",
    ),
):
    """
    Segment waveforms in a directory into syllabic segments and save the output as .pt files. Must have ffmpeg installed on your system.

    -----------------------------------------------------------------------------

    - If --output is [green]continuous[/green], save the continuous representations.
    Each output file is a dictionary with keys: "starts", "ends" and "embeddings".
    "starts" and "ends" are long tensors containing frame indices (divide by 50 Hz to get ).
    "embeddings" is a float32 tensor with one 1024-dimensional embedding per segment.

    - If --output is [green]discrete[/green] save the segments with a K-means cluster ID (one of 10,000 types) associated with each segment. One of 10,000 values.
    Each output file is a torch.long tensor with shape [num_segments, 3]. The columns are: start frame index, end frame index, cluster ID.

    - If --output is [green]collapsed[/green] save the segments with an K-means cluster ID (one of 9,116 types) associated with each segment, where all the centroids that correspond to silences are mapped to a single id, SIL=9115.
    Each output file is a torch.long tensor with shape [num_segments, 3]. The columns are: start frame index, end frame index, unit ID.

    -----------------------------------------------------------------------------

    """
    check_dependencies()

    from zerosyl.utils.encode import encode, encode_parallel

    if batch_size is None:
        # run single-threaded encoding

        encode(
            input_dir=input_dir,
            output_dir=output_dir,
            input_pattern=input_pattern,
            output_format=output_format,
            device=device,
        )
    else:
        # run parallel encoding
        encode_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            input_pattern=input_pattern,
            output_format=output_format,
            device=device,
            batch_size=batch_size,
            num_data_workers=num_data_workers,
            num_logic_workers=num_logic_workers,
        )


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
def loglikelihoods(
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
    normalize: bool = typer.Option(
        False, "--normalize", help="Normalize the loglikelihoods by token count"
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
        normalize=normalize,
    )


@eval_app.command()
def tsc(
    loglikelihoods_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to the log-likelihoods .txt file.",
    )
):
    """
    Compute the TSC metric from a log-likelihoods file.

    - The log-likelihoods file should be a text file where each line contains: <segment-file-stem> <log-likelihood>.
    """

    from zerosyl.eval.tsc import eval_tsc

    if loglikelihoods_file.suffix.lower() != ".txt":
        raise typer.BadParameter("loglikelihoods_file must have a .txt extension")

    score = eval_tsc(loglikelihoods_file)

    typer.echo(f"TSC Score: {score:.4f}")
