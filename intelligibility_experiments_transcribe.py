from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline
from whisper_normalizer.english import EnglishTextNormalizer


def transcribe_audio_files(
    input_dir: str | Path, output_dir: str | Path, model, normalizer
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    in_paths = list(input_dir.rglob("**/*.wav"))
    rel_paths = [
        wav_path.relative_to(input_dir).with_suffix(".txt") for wav_path in in_paths
    ]
    out_paths = [output_dir / rel_path for rel_path in rel_paths]

    # filter
    in_paths, out_paths = zip(
        *[
            (in_path, out_path)
            for in_path, out_path in zip(in_paths, out_paths)
            if not out_path.exists()
        ]
    )

    print(f"Found {len(out_paths)} audio files to process.")

    # for in_path in in_paths:
    #     print(in_path)

    for in_path, out_path in zip(tqdm(in_paths), out_paths):

        try:
            result = model(str(in_path))
            transcription_text = result["text"]
            normalized_text = normalizer(transcription_text)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(normalized_text)

        except Exception as e:
            print(f"Error processing file {in_path}: {e}\n")


# --- Main execution ---
if __name__ == "__main__":

    input_dir = "/home/nicolvisser/Workspace/zerosyl/output/synthesized"
    output_dir = "/home/nicolvisser/Workspace/zerosyl/output/transcribed"
    model_name = "openai/whisper-base"

    print(f"Loading Whisper model: {model_name}...")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )
    normalizer = EnglishTextNormalizer()
    print("Model loaded successfully.")

    transcribe_audio_files(input_dir, output_dir, transcriber, normalizer)
    print("Transcription finished.")
