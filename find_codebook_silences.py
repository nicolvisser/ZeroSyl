import torch
from scipy.cluster.hierarchy import cut_tree, linkage

centroids = torch.load("checkpoints/km10000-centroids-v040.pt").numpy()

linkage_matrix = linkage(
    centroids, method="ward", metric="euclidean", optimal_ordering=False
)

silences = cut_tree(linkage_matrix, 2)[:, 0]

# the smaller branch should be silences
if silences.sum() > (1 - silences).sum():
    silences = 1 - silences

silences = torch.from_numpy(silences).bool()

torch.save(silences, "checkpoints/km10000-silences-v040.pt")
