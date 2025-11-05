from pathlib import Path

import faiss
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from torchcodec.decoders import AudioDecoder
from tqdm.autonotebook import tqdm

from zerosyl.model import ZeroSylBase

checkpoint_path = Path("checkpoints/WavLM-Large.pt")
waveform_dir = Path("/mnt/wsl/newt/datasets/LibriSpeech")
output_dir = Path("checkpoints/kmeans-layer-13-win-3-prom_045")
num_clusters = 10000

assert checkpoint_path.exists()
assert waveform_dir.exists()
waveform_paths = list(waveform_dir.glob("train-clean-100/**/*.flac"))
assert len(waveform_paths) > 0

model = ZeroSylBase.from_pretrained_checkpoint(checkpoint_path).cuda()

all_embeddings = []

for waveform_path in tqdm(waveform_paths):
    decoder = AudioDecoder(waveform_path, sample_rate=16000, num_channels=1)
    audio = decoder.get_all_samples()
    embeddings, starts, ends = model.segment(audio.data.cuda())
    all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.concat(all_embeddings, axis=0)
num_points, num_dims = all_embeddings.shape
print(f"Found {num_points} points in {num_dims}D")

faiss.normalize_L2(all_embeddings)
init_centroids, _ = kmeans_plusplus(all_embeddings, num_clusters)
faiss.normalize_L2(init_centroids)

kmeans = faiss.Kmeans(
    d=init_centroids.shape[1],
    k=num_clusters,
    niter=100,
    verbose=True,
    spherical=True,
)
kmeans.train(all_embeddings, init_centroids=init_centroids)

Path("checkpoints").mkdir(parents=True, exist_ok=True)
np.save(f"checkpoints/km{num_clusters}-centroids.npy", kmeans.centroids)
np.save(f"checkpoints/km{num_clusters}-obj.npy", kmeans.obj)
torch.save(
    torch.from_numpy(kmeans.centroids), f"checkpoints/km{num_clusters}-centroids.npy"
)
