from pathlib import Path

import torch
from scipy.cluster.hierarchy import cut_tree, linkage


def find_and_save_codebook_silences(
    input_centroids_path: str | Path, output_silences_path: str | Path
):
    input_centroids_path = Path(input_centroids_path)
    output_silences_path = Path(output_silences_path)

    # load centroids
    centroids = torch.load(input_centroids_path)
    # agglomerative clustering
    linkage_matrix = linkage(
        centroids.numpy(), method="ward", metric="euclidean", optimal_ordering=False
    )
    # cut dendrogram into 2 main branches
    silences = cut_tree(linkage_matrix, 2)[:, 0]
    # the smaller branch should be silences
    if silences.sum() > (1 - silences).sum():
        silences = 1 - silences
    # to torch tensor
    silences = torch.from_numpy(silences).bool()
    # save
    torch.save(silences, output_silences_path)


if __name__ == "__main__":
    input_centroids_path = Path("checkpoints/km10000-centroids-v040.pt")
    output_silences_path = Path("checkpoints/km10000-silences-v040.pt")
    find_and_save_codebook_silences(input_centroids_path, output_silences_path)
