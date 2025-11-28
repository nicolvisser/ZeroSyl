from pathlib import Path

import torch
from IPython.display import Audio, display
from torchcodec.encoders import AudioEncoder

from zerosyl.acoustic import AcousticModel
from zerosyl.wavtokenizer.pretrained import load_wavtokenizer_small_600_24k_4096


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


checkpoint_path = Path("checkpoints/WavLM-Large.pt")
centroids_path = Path("checkpoints/km10000-centroids-v040.pt")

acoustic = AcousticModel.from_pretrained_checkpoint(
    "wandb/run-20251127_232329-37s9pfi1/files/step-20000.pt"
).cuda()
wavtokenizer = load_wavtokenizer_small_600_24k_4096().cuda()

segments_dir = Path(
    "output/segments/sylber-custom-centroids-with-silence-k-10001/LibriSpeech"
)
segments_paths = sorted(segments_dir.glob("dev*/**/*.pt"))

segments_paths = [
    segments_paths[100],
    segments_paths[101],
    segments_paths[0],
    segments_paths[1],
    segments_paths[30],
    segments_paths[31],
]

for segments_path in segments_paths:
    segments = torch.load(segments_path)

    tokens = acoustic.tokenize_infer(segments[:, 2])

    semantic_tokens_remaining = tokens[:-1].tolist()
    generated_ids = tokens.tolist()

    with torch.inference_mode():
        while True:
            current_token = generated_ids[-1]
            if current_token == acoustic.EOS:
                break
            if len(semantic_tokens_remaining) > 0 and (
                current_token == acoustic.BOS or current_token == acoustic.EOU
            ):
                generated_ids.append(semantic_tokens_remaining.pop(0))
            tokens = torch.tensor(generated_ids, dtype=torch.long, device="cuda")
            logits = acoustic.forward(tokens, [len(tokens)])
            next_token = sample(logits[-1], temperature=1.0, top_p=0.85).item()
            generated_ids.append(next_token)

    tokens = torch.tensor(generated_ids, dtype=torch.long, device="cuda")

    acoustic_units = acoustic.decode(tokens).cuda()
    bandwidth_id = torch.tensor([0], device="cuda")
    features = wavtokenizer.codes_to_features(acoustic_units.unsqueeze(0))
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).cpu()

    display(Audio(data=audio_out, rate=24000))

    Path("output/synth").mkdir(parents=True, exist_ok=True)
    AudioEncoder(audio_out, sample_rate=24000).to_file(
        f"output/synth/{segments_path.stem}-sylber.wav",
        sample_rate=24000,
        num_channels=1,
    )
