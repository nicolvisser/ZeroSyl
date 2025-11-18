from typing import List, Tuple

import faiss
import numpy as np
import torch
from scipy.signal import find_peaks

from .wavlm import WavLM, WavLMConfig

SAMPLE_RATE = 16000
FEATURE_RATE = 50.0


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
        # extract features
        wav = self._preprocess_waveform(wav)
        boundary_features, features_to_meanpool = self.extract_dual_features(
            wav, output_layer_1=self.boundary_layer, output_layer_2=self.meanpool_layer
        )
        boundary_features = boundary_features.squeeze(0)
        features_to_meanpool = features_to_meanpool.squeeze(0)

        # boundary detection
        peaks = self._promseg_on_norms(boundary_features)
        boundaries = [0] + peaks.tolist() + [len(boundary_features)]

        # meanpool within boundaries
        starts = torch.tensor(boundaries[:-1], device=wav.device)
        ends = torch.tensor(boundaries[1:], device=wav.device)
        embeddings = [
            features_to_meanpool[start:end].mean(dim=0)
            for start, end in zip(starts, ends)
        ]
        embeddings = torch.stack(embeddings)

        return embeddings, starts, ends

    # ------------------------- helper methods -------------------------

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
        assert wav.size(0) == 1
        if self.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        return wav

    def _promseg_on_norms(self, boundary_features: torch.Tensor) -> np.ndarray:
        """Perform prominence-based segmentation on feature norms.

        This method implements the core segmentation algorithm:
        1. Compute L2 norms of feature vectors across time
        2. Standardize norms to zero mean and unit variance
        3. Apply moving average smoothing with edge padding
        4. Detect peaks using scipy's find_peaks with prominence threshold

        Args:
            boundary_features: Feature tensor of shape (num_frames, model_dim).

        Returns:
            Array of frame indices where syllable boundaries are detected, excluding
            this first and last frame.
        """
        boundary_features = boundary_features.cpu().numpy()
        norms = np.linalg.norm(boundary_features, axis=-1)
        norms = (norms - norms.mean()) / norms.std()
        kernel = np.ones(self.window_size) / self.window_size
        pad_len = self.window_size // 2
        norms_padded = np.pad(norms, (pad_len, pad_len), mode="edge")
        norms_smooth = np.convolve(norms_padded, kernel, mode="valid")
        peaks, _ = find_peaks(norms_smooth, prominence=self.prominence)
        return peaks

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
        wav = self._preprocess_waveform(wav)
        boundary_features, _ = self.extract_features(
            wav, output_layer=self.boundary_layer
        )
        peaks = self._promseg_on_norms(boundary_features.squeeze(0))
        peaks_s = peaks / FEATURE_RATE
        duration = wav.size(-1) / SAMPLE_RATE
        boundaries_s = [0] + peaks_s.tolist() + [duration]
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
        embeddings, starts, ends = self.segment(wav)
        embeddings = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings)
        _, tokens = self.index.search(embeddings, 1)
        tokens = torch.from_numpy(tokens).squeeze().to(wav.device)
        return tokens, starts, ends
