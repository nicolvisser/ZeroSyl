"""ZeroSyl: boundary-aware segmentation and discrete syllable encoding.

This module provides three related models built on top of a WavLM
feature extractor:

- ``ZeroSylContinuous``: detects boundaries in continuous speech and
    mean-pools features within those boundaries to produce segment-level
    embeddings.
- ``ZeroSylDiscrete``: maps the segment-level embeddings to a
    discrete codebook (centroids) using FAISS nearest-neighbor search.
- ``ZeroSylCollapsed``: collapses multiple silence codes into a
    single silence class and merges adjacent silence segments.

These implementations are lightweight wrappers around a WavLM
extractor (see ``zerosyl.wavlm``) and are intended for use in
unsupervised syllable/segment discovery tasks and for producing
discrete representations suitable for downstream token-based
experiments such as segmentation and language modeling.

Key methods
-----------
- ``encode(wav)``: accepts a single-channel batched waveform of shape
    ``(1, num_samples)`` and returns ``(starts, ends, embeddings_or_ids)``.
    - For ``ZeroSylContinuous`` the third returned value is a tensor of
        embeddings with shape ``(num_segments, embed_dim)``.
    - For ``ZeroSylDiscrete``/``ZeroSylCollapsed`` the third returned
        value is a 1-D tensor of discrete ids (``int64``) with length
        ``num_segments``.

Usage example
-------------
>>> model = ZeroSylCollapsed.from_remote()
>>> wav = torch.randn(1, 16000)  # single second pseudo-waveform
>>> starts, ends, ids = model.encode(wav)

"""

from typing import Tuple

import faiss
import numpy as np
import torch
from scipy.signal import find_peaks

from .wavlm.WavLM import WavLM, WavLMConfig


class ZeroSylContinuous(WavLM):
    boundary_layer: int = 13
    window_size: int = 3
    prominence: float = 0.45
    meanpool_layer: int = 22

    sample_rate: int = 16000
    feature_rate: float = 50.0

    def __init__(self, wavlm_cfg: WavLMConfig):
        """Boundary-based continuous segmenter using WavLM features.

        Parameters
        ----------
        wavlm_cfg : WavLMConfig
            Configuration passed to the underlying WavLM feature extractor.
        """

        super().__init__(wavlm_cfg)

    @torch.inference_mode()
    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an input waveform into segment boundaries + embeddings.

        Parameters
        ----------
        wav : torch.Tensor
            Single-batch waveform tensor with shape ``(1, num_samples)``.

        Returns
        -------
        starts, ends, embeddings : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``starts``: 1-D tensor of start frame indices (feature frames).
            - ``ends``: 1-D tensor of end frame indices (feature frames).
            - ``embeddings``: 2-D tensor of shape ``(num_segments, embed_dim)``
              containing mean-pooled features for each detected segment.
        """

        # preprocess waveform
        assert wav.ndim == 2
        assert wav.size(0) == 1
        if self.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        # extract features
        boundary_features, semantic_features = self.extract_dual_features(
            wav, output_layer_1=self.boundary_layer, output_layer_2=self.meanpool_layer
        )
        boundary_features = boundary_features.squeeze(0)
        semantic_features = semantic_features.squeeze(0)

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
            semantic_features[start:end].mean(dim=0) for start, end in zip(starts, ends)
        ]
        embeddings = torch.stack(embeddings)

        return starts, ends, embeddings


class ZeroSylDiscrete(ZeroSylContinuous):
    def __init__(self, wavlm_cfg: WavLMConfig, centroids: torch.Tensor):
        """Discrete coder that maps segment embeddings to nearest centroids.

        Parameters
        ----------
        wavlm_cfg : WavLMConfig
            Configuration for the underlying WavLM extractor.
        centroids : torch.Tensor
            Tensor of cluster centroids with shape ``(K, D)`` where
            ``D`` must match ``cfg.encoder_embed_dim``.
        """

        super().__init__(wavlm_cfg)

        # store centroids as numpy array required by FAISS
        self.centroids = centroids.numpy()
        d = self.centroids.shape[1]
        assert d == self.cfg.encoder_embed_dim

        faiss.normalize_L2(self.centroids)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.centroids)

    @property
    def vocab_size(self) -> int:
        """Number of discrete centroids in the codebook."""
        return len(self.centroids)

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, centroids_path: str
    ) -> "ZeroSylDiscrete":
        """Load a model from local checkpoint files.

        Parameters
        ----------
        checkpoint_path : str
            Path to a saved WavLM checkpoint dict with keys ``cfg`` and
            ``model`` (state_dict).
        centroids_path : str
            Path to a tensor with centroids compatible with the model.

        Returns
        -------
        ZeroSylDiscrete
            An initialized and eval-mode model instance.
        """

        checkpoint = torch.load(checkpoint_path)
        centroids = torch.load(centroids_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    @classmethod
    def from_remote(cls) -> "ZeroSylDiscrete":
        """Download pretrained weights + centroids from remote storage.

        Returns an eval-mode ``ZeroSylDiscrete`` instance. Network
        weights and centroids are fetched using ``torch.hub`` utilities.
        """

        checkpoint = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/zerospeech-checkpoints/WavLM-Large.pt"
        )
        centroids = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040-centroids-k-10000.pt"
        )
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode waveform and map segment embeddings to discrete ids.

        Returns ``starts, ends, ids`` where ``ids`` is a 1-D tensor of
        nearest-centroid indices (dtype ``int64``) with length equal to
        the number of detected segments.
        """

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
        """Collapse multiple silence centroids into a single silence id.

        Parameters
        ----------
        wavlm_cfg : WavLMConfig
            WavLM configuration.
        centroids : torch.Tensor
            Centroids tensor passed to ``ZeroSylDiscrete``.
        silences : torch.Tensor
            Boolean-like tensor indicating which centroids correspond to
            silence. Must have same length as ``centroids``.
        """

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
        """Size of the collapsed vocabulary (silence mapped to a single id)."""
        return self.mapping.max().item() + 1

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, centroids_path: str, silences_path: str
    ) -> "ZeroSylCollapsed":
        """Load a collapsed model from local checkpoint + centroids + silences."""

        checkpoint = torch.load(checkpoint_path)
        centroids = torch.load(centroids_path)
        silences = torch.load(silences_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids, silences)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    @classmethod
    def from_remote(cls) -> "ZeroSylCollapsed":
        """Download pretrained weights, centroids and silence mask remotely."""

        checkpoint = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/zerospeech-checkpoints/WavLM-Large.pt"
        )
        centroids = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040-centroids-k-10000.pt"
        )
        silences = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040-silences-k-10000.pt"
        )
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids, silences)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def encode(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode waveform, collapse silence codes, and merge adjacent silences.

        After obtaining discrete ids from the parent class, this method
        (lazily) moves the internal ``mapping`` to the correct device,
        remaps ids so that all silence centroids share a single id, and
        merges consecutive segments that are the same silence id into a
        single longer segment.
        """

        starts, ends, ids = super().encode(wav)

        # lazily put mapping on the correct device
        if self.mapping.device != ids.device:
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

        return starts_merged, ends_merged, ids_merged
