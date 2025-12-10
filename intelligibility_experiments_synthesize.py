from pathlib import Path

import torch
from neucodec import DistillNeuCodec
from torchcodec.encoders import AudioEncoder

from zerosyl import AcousticModel

encoders = [
    "Sylber-k-10001",
    "SylBoost625-k-8192",
    "ZeroSylCollapsed-v040-k-9116",
]

neucodec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").cuda()

data_sizes = [100, 360, 460]

for encoder in encoders:
    for data_size in data_sizes:
        segments_dir = f"output/segments/{encoder}/LibriSpeech"
        segments_dir = Path(segments_dir)
        assert segments_dir.exists
        segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))
        assert len(segments_paths) > 0

        rel_paths = [
            p.relative_to(segments_dir).with_suffix(".wav") for p in segments_paths
        ]

        output_dir = f"output/synthesized/ELLA-V-{encoder}-neucodec-LibriSpeech-train-clean-{data_size}/LibriSpeech"
        output_paths = [output_dir / rel_path for rel_path in rel_paths]

        filter_fn = lambda x: not x[1].exists()
        in_out_paths = list(filter(filter_fn, zip(segments_paths, output_paths)))

        checkpoint_url = f"https://storage.googleapis.com/zerospeech-checkpoints/ELLA-V-{encoder}-neucodec-LibriSpeech-train-clean-{data_size}.pt"
        acoustic_model = AcousticModel.from_remote(checkpoint_url).cuda()

        max_batch_size = 100

        print(f"Segments directory: {segments_dir}")
        print(f"Number of segments to be processed: {len(in_out_paths)}")
        print(f"Checkpoint URL: {checkpoint_url}")
        print(f"Batch size: {max_batch_size}")
        print(f"Processing...")

        for first_item_idx in range(0, len(in_out_paths), max_batch_size):
            batch_in_out_paths = in_out_paths[
                first_item_idx : first_item_idx + max_batch_size
            ]
            batch_in_paths, batch_out_paths = zip(*batch_in_out_paths)

            print(
                f"  {first_item_idx:06d} -> {first_item_idx + len(batch_in_out_paths):06d}"
            )

            segments_list = [torch.load(p) for p in batch_in_paths]
            semantic_units_list = [s[:, 2].cuda() for s in segments_list]

            acoustic_units_list, finished_list = acoustic_model.generate(
                semantic_units_list,
                temperature=1.0,
                top_p=0.85,
                max_tokens_per_semantic_unit=20,
                max_tokens=2000,
                show_progress=True,
                return_finished_list=True,
            )

            for acoustic_units, finished, out_path in zip(
                acoustic_units_list, finished_list, batch_out_paths, strict=True
            ):

                with torch.inference_mode():
                    waveform = (
                        neucodec.decode_code(acoustic_units[None, None, :])
                        .squeeze(0)
                        .cpu()
                    )

                out_path.parent.mkdir(parents=True, exist_ok=True)
                AudioEncoder(waveform, sample_rate=24000).to_file(out_path)

            torch.cuda.empty_cache()
