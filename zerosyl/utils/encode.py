from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import torch
import torch.multiprocessing as mp
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder

from ..zerosyl import WavLM, ZeroSylCollapsed, ZeroSylContinuous, ZeroSylDiscrete

# =========================== SINGLE-THREADED ENCODING FUNCTION ===========================


def encode(
    input_dir: str | Path,
    output_dir: str | Path,
    input_pattern: str = "*.wav",
    output_format: Literal["continuous", "discrete", "collapsed"] = "collapsed",
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Segment a directory of waveforms into syllabic segments and save the output as .pt files.

    This function runs in a single thread and might be slow for large datasets. For faster encoding, consider using the `encode_parallel` function instead.

    -----------------------------------------------------------------------------

    - If --output is "continuous", save the continuous representations.
    Each output file is a dictionary with keys: "starts", "ends" and "embeddings".
    "starts" and "ends" are long tensors containing frame indices.
    "embeddings" is a float32 tensor with one 1024-dimensional embedding per segment.

    - If --output is "discrete" save the segments with a K-means cluster ID (one of 10,000 types) associated with each segment.

    - If --output is "collapsed" save the segments with an K-means cluster ID (one of 9,116 types) associated with each segment, where all the centroids that correspond to silences are mapped to a single id, SIL=9115.

    -----------------------------------------------------------------------------

    """
    input_dir = Path(input_dir)
    input_paths = list(input_dir.rglob(input_pattern))

    assert (
        len(input_paths) > 0
    ), f"No files found in {input_dir} matching pattern {input_pattern}"

    if output_format == "continuous":
        zerosyl_continuous = ZeroSylContinuous.from_remote().to(device)
    elif output_format == "discrete":
        zerosyl_discrete = ZeroSylDiscrete.from_remote().to(device)
    else:  # output_format == "collapsed"
        zerosyl_collapsed = ZeroSylCollapsed.from_remote().to(device)

    for input_path in track(input_paths, description="Encoding files..."):
        rel_path = input_path.relative_to(input_dir)
        output_path = Path(output_dir) / rel_path.with_suffix(".pt")

        decoder = AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data.to(device)

        if output_format == "continuous":
            starts, ends, embeddings = zerosyl_continuous.encode(wav)
            output_data = {"starts": starts, "ends": ends, "embeddings": embeddings}
        elif output_format == "discrete":
            starts, ends, ids = zerosyl_discrete.encode(wav)
            output_data = torch.stack([starts, ends, ids], dim=1).cpu()
        elif output_format == "collapsed":
            starts, ends, ids = zerosyl_collapsed.encode(wav)
            output_data = torch.stack([starts, ends, ids], dim=1).cpu()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_data, output_path)


# =========================== PARALLEL ENCODING FUNCTION ===========================

# --- 1. WORKER FUNCTIONS ---


def _logic_worker_fn(
    worker_id,
    input_queue,
    output_queue,
    output_format: Literal["continuous", "discrete", "collapsed"] = "collapsed",
    window_size: int = 3,
    prominence: float = 0.45,
):

    # initialize FAISS index and centroids inside the worker
    centroids = torch.hub.load_state_dict_from_url(
        "https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040-centroids-k-10000.pt"
    )
    centroids = centroids.numpy()
    d = centroids.shape[1]
    faiss.normalize_L2(centroids)
    index = faiss.IndexFlatIP(d)
    index.add(centroids)

    # initialize stuff for remapping silences
    silences = torch.hub.load_state_dict_from_url(
        "https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040-silences-k-10000.pt"
    )
    # create a mapping that will merge all the silence (=True) entries
    # while placing all the non-silences at the start of the codebook
    # and placing the single silence at the end of the codebook
    order = torch.argsort(silences)  # [0,0,....,0,0,1,1,...,1,1]
    SIL = torch.argmax(silences[order].long()).item()  # position of first 1
    mapping = torch.empty_like(order)
    mapping[order] = torch.arange(len(order))
    mapping[mapping > SIL] = SIL

    while True:
        try:
            item = input_queue.get()
            if item is None:
                break

            rel_path, boundary_features, semantic_features = item

            # boundary detection
            boundary_features = boundary_features.cpu().numpy()
            norms = np.linalg.norm(boundary_features, axis=-1)
            norms = (norms - norms.mean()) / norms.std()
            kernel = np.ones(window_size) / window_size
            pad_len = window_size // 2
            norms_padded = np.pad(norms, (pad_len, pad_len), mode="edge")
            norms_smooth = np.convolve(norms_padded, kernel, mode="valid")
            peaks, _ = find_peaks(norms_smooth, prominence=prominence)
            boundaries = [0] + peaks.tolist() + [len(boundary_features)]

            # meanpool within boundaries
            starts = torch.tensor(boundaries[:-1])
            ends = torch.tensor(boundaries[1:])
            embeddings = [
                semantic_features[start:end].mean(dim=0)
                for start, end in zip(starts, ends)
            ]
            embeddings = torch.stack(embeddings)

            if output_format == "continuous":
                data_to_save = {
                    "starts": starts,
                    "ends": ends,
                    "embeddings": embeddings,
                }
                output_queue.put((rel_path, data_to_save))
                continue

            # k-means assignment
            embeddings = embeddings.cpu().numpy()
            faiss.normalize_L2(embeddings)
            _, ids = index.search(embeddings, 1)
            ids = torch.from_numpy(ids).long().squeeze()

            if output_format == "discrete":
                data_to_save = torch.stack([starts, ends, ids], dim=1)
                output_queue.put((rel_path, data_to_save))
                continue

            # remap such that there is only one silence type
            ids = mapping[ids]

            # merge duplicate SIL segments
            not_repeated = torch.ones_like(ids, dtype=torch.bool)
            not_repeated[1:] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == SIL)
            is_end = torch.ones_like(ids, dtype=torch.bool)
            is_end[:-1] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == SIL)
            starts_merged = starts[not_repeated]
            ends_merged = ends[is_end]
            ids_merged = ids[not_repeated]

            data_to_save = torch.stack([starts_merged, ends_merged, ids_merged], dim=1)
            output_queue.put((rel_path, data_to_save))

        except Exception as e:
            print(f"Error in logic worker {worker_id}: {e}")


def _io_worker_fn(output_queue, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            item = output_queue.get()
            if item is None:
                break
            rel_path, data_to_save = item
            save_path = output_dir / rel_path.with_suffix(".pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data_to_save, save_path)

        except Exception as e:
            print(f"Error in IO worker: {e}")


class WavDataset(Dataset):
    def __init__(self, wav_dir: str | Path, wav_pattern: str = "*.wav"):
        self.wav_dir = Path(wav_dir)
        self.wav_paths = list(self.wav_dir.glob(wav_pattern))
        assert (
            len(self.wav_paths) > 0
        ), f"No files found in {wav_dir} matching pattern {wav_pattern}"
        self.wav_paths = sorted(
            self.wav_paths, key=lambda x: x.stat().st_size, reverse=True
        )

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        rel_path = wav_path.relative_to(self.wav_dir)
        decoder = AudioDecoder(wav_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
        pad_length = 320 - (wav.size(-1) % 320)
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        seqlen = wav.size(-1) // 320
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        wav = wav.squeeze(0)
        return rel_path, wav, seqlen

    def collate_fn(self, batch):
        rel_paths, wavs, seqlens = zip(*batch)
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0)
        return rel_paths, wavs, seqlens


# --- 2. MAIN EXECUTION ---


def encode_parallel(
    input_dir: str | Path,
    output_dir: str | Path,
    input_pattern: str = "lexical/dev/*.wav",
    output_format: Literal["continuous", "discrete", "collapsed"] = "collapsed",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 8,
    num_data_workers: int = 4,
    num_logic_workers: int = 1,
):
    """
    Executes a high-throughput, multiprocessed pipeline to encode audio files into syllabic segments.

    This function CPU-intensive segmentation logic and disk I/O to separate processes. The pipeline
    operates in four stages:
    1. **Data Loading:** Fetches and pads audio waveforms (via `num_data_workers`).
    2. **Inference (Main Process):** Runs WavLM on the GPU (batched) to extract frame-level boundary and semantic features.
    3. **Logic Processing (Workers):** Consumes raw features, performs peak detection (boundary discovery),
       and pools semantic features into segment-level embeddings.
    4. **I/O (Worker):** Asynchronously saves the resulting tensors to disk.

    Args:
        wav_dir (str | Path): The root directory containing the source WAV files.
        output_dir (str | Path): The destination directory where processed `.pt` files will be saved,
            mirroring the relative structure of `wav_dir`.
        wav_pattern (str, optional): A glob pattern to filter specific files within `wav_dir`.
            Defaults to "lexical/dev/*.wav".
        batch_size (int, optional): The number of audio files to process in a single GPU inference step.
            Defaults to 16.
        num_logic_workers (int | None, optional): The number of parallel CPU processes dedicated to
            segmentation logic (peak finding and mean pooling). If None, it is calculated based
            on available CPUs.
        num_data_workers (int | None, optional): The number of workers used by the PyTorch DataLoader
            for fetching audio files. If None, it is calculated based on available CPUs.
        device (torch.device, optional): The device to run the WavLM model on.
            Defaults to "cuda" if available, otherwise "cpu".

    Returns:
        None
            The function saves files to `output_dir`. Each saved `.pt` file contains a dictionary with:
            - **starts** (Tensor): Start indices of detected segments.
            - **ends** (Tensor): End indices of detected segments.
            - **embeddings** (Tensor): Mean-pooled semantic features for each segment.
    """

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    num_available_cpus = mp.cpu_count()
    print(f"Available CPUs: {num_available_cpus}")

    assert num_logic_workers + num_data_workers + 2 <= num_available_cpus, (
        f"Not enough CPUs for the requested number of workers. "
        f"Requested: {num_logic_workers} logic workers + {num_data_workers} data workers + 1 main thread + 1 IO worker. "
        f"Available CPUs: {num_available_cpus}."
    )

    print(
        f"Using {num_data_workers} data workers, 1 main thread + GPU, {num_logic_workers} logic workers, and 1 IO worker."
    )

    print(
        "Multiprocessing will have some overhead, please only use on large datasets and try different batch sizes and worker counts."
    )

    print(
        "Time estimates will not be accurate, because larger batches are processed first. It will likely finish faster than estimated."
    )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # dataset and loader
    dataset = WavDataset(input_dir, input_pattern)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_data_workers,
    )

    # model
    wavlm = WavLM.from_remote().to(device)
    wavlm.eval()

    # queues
    feature_queue = mp.Queue(maxsize=32)
    save_queue = mp.Queue(maxsize=32)

    # start workers
    io_process = mp.Process(target=_io_worker_fn, args=(save_queue, output_dir))
    io_process.start()
    workers = []
    for i in range(num_logic_workers):
        p = mp.Process(
            target=_logic_worker_fn,
            args=(i, feature_queue, save_queue, output_format),
        )
        p.start()
        workers.append(p)

    # main loop: run (GPU) inference and put features in queue
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:

        # Add your task here
        task_workers = progress.add_task("Preparing workers...", total=1)
        task_encoding = None

        try:
            with torch.no_grad():
                for rel_paths, wavs, seqlens in loader:
                    max_seqlen = max(seqlens)

                    padding_mask = torch.arange(max_seqlen, device=device).unsqueeze(
                        0
                    ) >= torch.tensor(seqlens, device=device).unsqueeze(1)

                    boundary_features, semantic_features = wavlm.extract_dual_features(
                        wavs.to(device), padding_mask=padding_mask
                    )

                    boundary_features = boundary_features.cpu()
                    semantic_features = semantic_features.cpu()

                    for rp, bf, sf, sl in zip(
                        rel_paths, boundary_features, semantic_features, seqlens
                    ):
                        bf_sliced = bf[:sl].clone()
                        sf_sliced = sf[:sl].clone()
                        feature_queue.put((rp, bf_sliced, sf_sliced))

                    if task_encoding is None:
                        progress.update(task_workers, completed=1)
                        task_encoding = progress.add_task(
                            "Encoding...", total=len(loader)
                        )

                    progress.update(task_encoding, advance=1)

        except torch.OutOfMemoryError:
            print("Ran out of GPU memory. Consider reducing the batch size...")

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            print("GPU finished. Waiting for workers to clear queues...")

            # signal to workers to shut down
            for _ in range(num_logic_workers):
                feature_queue.put(None)

            for p in workers:
                p.join()

            save_queue.put(None)
            io_process.join()

            print("Done!")
