from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from ..encoder import EncodecModel
from ..encoder.modules import SEANetDecoder, SEANetEncoder
from ..encoder.quantization import ResidualVectorQuantizer


@dataclass
class EncodecFeaturesArgs:
    encodec_model: str = "encodec_24khz"
    bandwidths: List[float] = field(default_factory=lambda: [6.6, 6.6, 6.6, 6.6])
    train_codebooks: bool = True
    num_quantizers: int = 1
    downsamples: List[int] = field(default_factory=lambda: [8, 5, 4, 2])
    vq_bins: int = 4096
    vq_kmeans: int = 200


class EncodecFeatures(nn.Module):
    def __init__(self, args: EncodecFeaturesArgs):
        super().__init__()

        self.frame_rate = 25

        encoder = SEANetEncoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=args.downsamples,
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        decoder = SEANetDecoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=[8, 5, 4, 2],
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        quantizer = ResidualVectorQuantizer(
            dimension=512,
            n_q=args.num_quantizers,
            bins=args.vq_bins,
            kmeans_iters=args.vq_kmeans,
            decay=0.99,
            kmeans_init=True,
        )

        if args.encodec_model == "encodec_24khz":
            self.encodec = EncodecModel(
                encoder=encoder,
                decoder=decoder,
                quantizer=quantizer,
                target_bandwidths=args.bandwidths,
                sample_rate=24000,
                channels=1,
            )
        else:
            raise ValueError(
                f"Unsupported encodec_model: {args.encodec_model}. Supported options are 'encodec_24khz'."
            )
        for param in self.encodec.parameters():
            param.requires_grad = True

        self.bandwidths = args.bandwidths

    def forward(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)

        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer(
            emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
        )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty

        return quantized, codes, commit_loss

    def infer(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer.infer(
            emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
        )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty

        return quantized, codes, commit_loss
