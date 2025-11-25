from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .feature_extractors import EncodecFeatures, EncodecFeaturesArgs
from .heads import ISTFTHead, ISTFTHeadArgs
from .models import VocosBackbone, VocosBackboneArgs


@dataclass
class WavTokenizerArgs:
    feature_extractor: EncodecFeaturesArgs = field(default_factory=EncodecFeaturesArgs)
    backbone: VocosBackboneArgs = field(default_factory=VocosBackboneArgs)
    head: ISTFTHeadArgs = field(default_factory=ISTFTHeadArgs)


class WavTokenizer(nn.Module):
    """
    The WavTokenizer class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self,
        args: WavTokenizerArgs,
    ):
        super().__init__()
        self.feature_extractor = EncodecFeatures(args.feature_extractor)
        self.backbone = VocosBackbone(args.backbone)
        self.head = ISTFTHead(args.head)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, args: WavTokenizerArgs, model_path: str) -> "WavTokenizer":
        model = cls(args)
        state_dict_raw = torch.load(model_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v

        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features, _, _ = self.feature_extractor(audio_input, **kwargs)  # 0818
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def encode(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        features, discrete_codes, _ = self.feature_extractor.infer(
            audio_input, **kwargs
        )
        return features, discrete_codes

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    @torch.inference_mode()
    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
        codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (K, L) or (K, B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        assert isinstance(
            self.feature_extractor, EncodecFeatures
        ), "Feature extractor should be an instance of EncodecFeatures"

        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        tmp = torch.cat(
            [vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers],
            dim=0,
        )
        # features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)

        return features
