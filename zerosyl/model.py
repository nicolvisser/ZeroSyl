from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
from scipy.signal import find_peaks

from .wavlm import WavLM, WavLMConfig

SAMPLE_RATE = 16000
FEATURE_RATE = 50.0
DOWNSAMPLE_FACTOR = 320


class ZeroSylBase(WavLM):
    """Zero-resource syllable segmentation model based on WavLM.

    This class extends WavLM to perform unsupervised syllable boundary detection
    using prominence-based segmentation on feature norms, followed by meanpooling
    to generate syllable-level embeddings.

    Attributes:
        boundary_layer: Transformer layer number from which to extract features
            for boundary detection. Default is 11.
        window_size: Size of the moving average window for smoothing feature norms
            before peak detection. Default is 3.
        prominence: Minimum prominence threshold for peak detection in the smoothed
            norm signal. Higher values result in fewer detected boundaries. Default is 0.5.
        meanpool_layer: Optional layer number for extracting features to meanpool.
            If None, uses the final output layer of WavLM. Default is None.
    """

    boundary_layer: int = 13  # to compute norms
    window_size: int = 3  # for norm smoothing
    prominence: float = 0.45  # for peak detection
    meanpool_layer: int = 22  # layer to meanpool

    # ------------------------- core methods -------------------------

    @torch.inference_mode()
    def segment(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment audio into syllable-like units and compute meanpooled embeddings.

        This is the core segmentation method that combines boundary detection with
        feature extraction to produce one embedding per detected segment.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            A tuple containing:
                - embeddings: Tensor of shape (num_segments, model_dim) containing
                  one meanpooled embedding per detected segment.
                - starts: Tensor of shape (num_segments,) with frame indices of
                  segment start boundaries.
                - ends: Tensor of shape (num_segments,) with frame indices of
                  segment end boundaries.
        """
        return self.segment_batch([wav])[0]

    @torch.inference_mode()
    def segment_batch(
        self, wavs: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Segment audio into syllable-like units and compute meanpooled embeddings.

        This is the core segmentation method that combines boundary detection with
        feature extraction to produce one embedding per detected segment.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            A tuple containing:
                - embeddings: Tensor of shape (num_segments, model_dim) containing
                  one meanpooled embedding per detected segment.
                - starts: Tensor of shape (num_segments,) with frame indices of
                  segment start boundaries.
                - ends: Tensor of shape (num_segments,) with frame indices of
                  segment end boundaries.
        """
        wavs, seqlens, padding_mask = self._pad_and_stack(wavs)

        # extract features
        wavs = self._preprocess_waveform(wavs)
        boundary_features, features_to_meanpool = self.extract_dual_features(
            wavs,
            padding_mask,
            output_layer_1=self.boundary_layer,
            output_layer_2=self.meanpool_layer,
        )

        # boundary detection
        norms = self._norms(boundary_features)
        norms = norms.cpu().numpy()

        output = []
        for n, l, f in zip(norms, seqlens, boundary_features):
            peaks, _ = find_peaks(n[:l], prominence=self.prominence)
            boundaries = [0] + peaks.tolist() + [l]

            # meanpool within boundaries
            starts = torch.tensor(boundaries[:-1], device=wavs.device)
            ends = torch.tensor(boundaries[1:], device=wavs.device)
            embeddings = [f[start:end].mean(dim=0) for start, end in zip(starts, ends)]
            embeddings = torch.stack(embeddings)
            output.append((embeddings, starts, ends))

        return output

    # ------------------------- helper methods -------------------------
    def _pad_and_stack(self, wavs: List[torch.Tensor]):
        wavs = [
            wav[0, : wav.size(-1) // DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR]
            for wav in wavs
        ]
        seqlens = [wav.size(-1) // DOWNSAMPLE_FACTOR for wav in wavs]
        seqlens = torch.tensor(seqlens, device=wavs[0].device)
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        indices = torch.arange(seqlens.max(), device=wavs.device)
        padding_mask = indices.unsqueeze(0) >= seqlens.unsqueeze(1)
        return wavs, seqlens, padding_mask

    def _preprocess_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Preprocess waveform for WavLM compatibility.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            Normalized and padded waveform tensor with 40 samples added to each side.

        Raises:
            AssertionError: If wav is not 2D or batch size is not 1.
        """
        assert wav.ndim == 2
        bsz, seqlen = wav.shape
        if self.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, (seqlen,))
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        return wav

    def _norms(self, boundary_features: torch.Tensor) -> List[torch.Tensor]:
        assert boundary_features.ndim == 3  # (bsz, seqlen, model_dim)
        norms = torch.linalg.vector_norm(boundary_features, dim=-1)
        norms = (norms - norms.mean(dim=1, keepdim=True)) / norms.std(
            dim=1, keepdim=True
        )
        kernel = (
            torch.ones(self.window_size, dtype=torch.float32, device=norms.device)
            / self.window_size
        )
        norms_smooth = torch.nn.functional.conv1d(
            norms.unsqueeze(1), kernel.view(1, 1, -1), padding=3 // 2
        )
        norms_smooth = norms_smooth.squeeze(1)  # (bsz, seqlen)
        return norms_smooth

    # ------------------------- convenience methods -------------------------

    @torch.inference_mode()
    def boundaries(self, wav: torch.Tensor) -> List[float]:
        """Extract syllable boundary timestamps in seconds.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            List of boundary timestamps in seconds, including 0.0 at the start
            and the audio duration at the end.
        """
        return self.boundaries_batch([wav])[0]

    @torch.inference_mode()
    def boundaries_batch(self, wavs: List[torch.Tensor]) -> List[List[float]]:
        """Extract syllable boundary timestamps in seconds.

        Args:
            wav: List[torch.Tensor] of input waveform tensors of shape (1, num_samples).

        Returns:
            List of boundary timestamps in seconds, including 0.0 at the start
            and the audio duration at the end.
        """
        wavs, seqlens, padding_mask = self._pad_and_stack(wavs)
        wavs = self._preprocess_waveform(wavs)
        boundary_features, _ = self.extract_features(
            wavs, padding_mask, output_layer=self.boundary_layer
        )  # (bsz,seqlen,model_dim)
        norms = self._norms(boundary_features)
        norms = norms.cpu().numpy()
        boundaries_s = []
        for n, l in zip(norms, seqlens):
            peaks, _ = find_peaks(n[:l], prominence=self.prominence)
            peaks_s = peaks / FEATURE_RATE
            duration = wavs.size(-1) / SAMPLE_RATE
            boundaries_s.append([0] + peaks_s.tolist() + [duration])
        return boundaries_s


class ZeroSylDiscrete(ZeroSylBase):
    """Zero-resource syllable segmentation model with discrete token quantization.

    This class extends ZeroSylBase to add vector quantization capabilities, mapping
    continuous syllable embeddings to discrete tokens using k-means centroids and
    FAISS for efficient nearest neighbor search.

    Attributes:
        centroids: NumPy array of shape (vocab_size, model_dim) containing the
            L2-normalized k-means centroids used for quantization.
        vocab_size: Number of discrete tokens in the vocabulary.
        index: FAISS IndexFlatIP index for efficient nearest centroid search
            using inner product (equivalent to cosine similarity after normalization).
    """

    def __init__(self, wavlm_cfg: WavLMConfig, centroids: torch.Tensor):
        """Initialize ZeroSylDiscrete with WavLM configuration and quantization centroids.

        Args:
            wavlm_cfg: WavLM configuration object specifying model architecture.
            centroids: Tensor of shape (vocab_size, model_dim) containing k-means
                centroids for vector quantization. Must match the model's embedding
                dimension.

        Raises:
            AssertionError: If centroid dimension doesn't match encoder_embed_dim.
        """
        super().__init__(wavlm_cfg)
        self.centroids = centroids.numpy()
        self.vocab_size, d = self.centroids.shape
        assert d == self.cfg.encoder_embed_dim
        faiss.normalize_L2(self.centroids)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.centroids)

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, centroids_path: str
    ) -> "ZeroSylDiscrete":
        """Load a pretrained ZeroSylDiscrete model from checkpoint files.

        Args:
            checkpoint_path: Path to the WavLM model checkpoint file containing
                configuration and model state dict.
            centroids_path: Path to the .pt file containing the k-means centroids
                for quantization.

        Returns:
            Initialized ZeroSylDiscrete model in evaluation mode with loaded
            weights and centroids.
        """
        checkpoint = torch.load(checkpoint_path)
        centroids = torch.load(centroids_path)
        cfg = WavLMConfig(checkpoint["cfg"])
        model = cls(cfg, centroids)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def tokenize(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment audio and quantize syllable embeddings to discrete tokens.

        This method combines syllable segmentation with vector quantization to
        produce discrete token sequences. Each detected syllable segment is mapped
        to its nearest centroid using cosine similarity.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            A tuple containing:
                - tokens: Tensor of shape (num_segments,) containing discrete token
                    IDs (indices into the centroid vocabulary) for each segment.
                - starts: Tensor of shape (num_segments,) with frame indices of
                    segment start boundaries.
                - ends: Tensor of shape (num_segments,) with frame indices of
                    segment end boundaries.
        """
        return self.tokenize_batch([wav])[0]

    def tokenize_batch(
        self, wavs: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Segment audio and quantize syllable embeddings to discrete tokens.

        This method combines syllable segmentation with vector quantization to
        produce discrete token sequences. Each detected syllable segment is mapped
        to its nearest centroid using cosine similarity.

        Args:
            wav: Input waveform tensor of shape (1, num_samples).

        Returns:
            A tuple containing:
                - tokens: Tensor of shape (num_segments,) containing discrete token
                    IDs (indices into the centroid vocabulary) for each segment.
                - starts: Tensor of shape (num_segments,) with frame indices of
                    segment start boundaries.
                - ends: Tensor of shape (num_segments,) with frame indices of
                    segment end boundaries.
        """
        outputs = []
        for embeddings, starts, ends in self.segment_batch(wavs):
            embeddings = embeddings.cpu().numpy()
            faiss.normalize_L2(embeddings)
            _, tokens = self.index.search(embeddings, 1)
            tokens = torch.from_numpy(tokens).squeeze().to(starts.device)
            outputs.append((tokens, starts, ends))
        return outputs
