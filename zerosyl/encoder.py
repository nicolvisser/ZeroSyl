from functools import reduce
from typing import Tuple

import faiss
import numpy as np
import torch
from scipy.signal import find_peaks

from .wavlm import WavLM, WavLMConfig


class ZeroSylContinuous(WavLM):
    boundary_layer: int = 13
    window_size: int = 3
    prominence: float = 0.45
    meanpool_layer: int = 22

    sample_rate: int = 16000
    feature_rate: float = 50.0

    def __init__(self, wavlm_cfg: WavLMConfig):
        super().__init__(wavlm_cfg)

    @torch.inference_mode()
    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # preprocess waveform
        assert wav.ndim == 2
        assert wav.size(0) == 1
        if self.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        # extract features
        boundary_features, features_to_meanpool = self.extract_dual_features(
            wav, output_layer_1=self.boundary_layer, output_layer_2=self.meanpool_layer
        )
        boundary_features = boundary_features.squeeze(0)
        features_to_meanpool = features_to_meanpool.squeeze(0)

        # boundary detection
        boundary_features = boundary_features.cpu().numpy()
        norms = np.linalg.norm(boundary_features, axis=-1)
        norms = (norms - norms.mean()) / norms.std()
        kernel = np.ones(self.window_size) / self.window_size
        pad_len = self.window_size // 2
        norms_padded = np.pad(norms, (pad_len, pad_len), mode="edge")
        norms_smooth = np.convolve(norms_padded, kernel, mode="valid")
        peaks, _ = find_peaks(norms_smooth, prominence=self.prominence)
        boundaries = [0] + peaks.tolist() + [len(boundary_features)]

        # meanpool within boundaries
        starts = torch.tensor(boundaries[:-1], device=wav.device)
        ends = torch.tensor(boundaries[1:], device=wav.device)
        embeddings = [
            features_to_meanpool[start:end].mean(dim=0)
            for start, end in zip(starts, ends)
        ]
        embeddings = torch.stack(embeddings)

        return starts, ends, embeddings


class ZeroSylDiscrete(ZeroSylContinuous):
    def __init__(self, wavlm_cfg: WavLMConfig, centroids: torch.Tensor):
        super().__init__(wavlm_cfg)

        self.centroids = centroids.numpy()
        d = self.centroids.shape[1]
        assert d == self.cfg.encoder_embed_dim

        faiss.normalize_L2(self.centroids)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.centroids)

    @property
    def vocab_size(self) -> int:
        return len(self.centroids)

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, centroids_path: str
    ) -> "ZeroSylDiscrete":
        checkpoint = torch.load(checkpoint_path)
        centroids = torch.load(centroids_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts, ends, embeddings = super().encode(wav)

        embeddings = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings)
        _, ids = self.index.search(embeddings, 1)
        ids = torch.from_numpy(ids).squeeze().to(wav.device)
        return starts, ends, ids


class ZeroSylCollapsed(ZeroSylDiscrete):

    def __init__(
        self, wavlm_cfg: WavLMConfig, centroids: torch.Tensor, silences: torch.Tensor
    ):
        super().__init__(wavlm_cfg, centroids)

        assert len(silences) == len(centroids)

        # create a mapping that will merge all the silence (=True) entries
        # while placing all the non-silences at the start of the codebook
        # and placing the single silence at the end of the codebook
        order = torch.argsort(silences)  # [0,0,....,0,0,1,1,...,1,1]
        self.SIL = torch.argmax(silences[order].long()).item()  # position of first 1
        self.mapping = torch.empty_like(order)
        self.mapping[order] = torch.arange(len(order))
        self.mapping[self.mapping > self.SIL] = self.SIL

    @property
    def vocab_size(self) -> int:
        return self.mapping.max().item() + 1

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, centroids_path: str, silences_path: str
    ) -> "ZeroSylCollapsed":
        checkpoint = torch.load(checkpoint_path)
        centroids = torch.load(centroids_path)
        silences = torch.load(silences_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids, silences)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts, ends, ids = super().encode(wav)

        if self.mapping.device != ids.device:
            # lazily put mapping on the correct device
            self.mapping = self.mapping.to(ids.device)

        # remap such that there is only one silence type
        ids = self.mapping[ids]

        # merge duplicate SIL segments
        not_repeated = torch.ones_like(ids, dtype=torch.bool)
        not_repeated[1:] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == self.SIL)
        is_end = torch.ones_like(ids, dtype=torch.bool)
        is_end[:-1] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == self.SIL)
        starts_merged = starts[not_repeated]
        ends_merged = ends[is_end]
        ids_merged = ids[not_repeated]
        assert len(starts_merged) == len(ends_merged) == len(ids_merged)

        return starts_merged, ends_merged, ids_merged
