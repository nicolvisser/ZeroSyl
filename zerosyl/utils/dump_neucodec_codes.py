from pathlib import Path

import torch
from neucodec import DistillNeuCodec
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


def dump_codes(
    wav_dir: str | Path, out_dir: str | Path, wav_pattern: str = "**/*.flac"
):
    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)

    model = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").cuda()
    model.eval()

    wav_paths = list(wav_dir.glob(wav_pattern))

    for wav_path in tqdm(wav_paths):

        decoder = AudioDecoder(wav_path, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples()

        with torch.no_grad():
            fsq_codes = model.encode_code(audio.data.unsqueeze(0).cuda())
        fsq_codes = fsq_codes.squeeze(0).squeeze(0).to(dtype=torch.uint16, device="cpu")

        rel_path = wav_path.relative_to(wav_dir).with_suffix(".pt")
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(fsq_codes, out_path)


if __name__ == "__main__":
    wav_dir = "/home/nicolvisser/Data/waveforms/LibriSpeech"
    out_dir = "output/acoustic_units/distill-neucodec/LibriSpeech"
    wav_pattern = "train-clean*/**/*.flac"
    dump_codes(wav_dir, out_dir, wav_pattern)
