from .decoder.pretrained import WavTokenizerArgs
from .decoder.feature_extractors import EncodecFeaturesArgs
from .decoder.heads import ISTFTHeadArgs
from .decoder.models import VocosBackboneArgs

ARGS_SMALL_600_24K_4096 = WavTokenizerArgs(
    feature_extractor=EncodecFeaturesArgs(
        encodec_model="encodec_24khz",
        bandwidths=[6.6, 6.6, 6.6, 6.6],
        train_codebooks=True,
        num_quantizers=1,
        downsamples=[6, 5, 5, 4],
        vq_bins=4096,
        vq_kmeans=200,
    ),
    backbone=VocosBackboneArgs(
        input_channels=512,
        dim=768,
        intermediate_dim=2304,
        num_layers=12,
        adanorm_num_embeddings=4,
    ),
    head=ISTFTHeadArgs(
        dim=768,
        n_fft=2400,
        hop_length=600,
        padding="same",
    ),
)

ARGS_SMALL_320_24K_4096 = WavTokenizerArgs(
    feature_extractor=EncodecFeaturesArgs(
        encodec_model="encodec_24khz",
        bandwidths=[6.6, 6.6, 6.6, 6.6],
        train_codebooks=True,
        num_quantizers=1,
        downsamples=[8, 5, 4, 2],
        vq_bins=4096,
        vq_kmeans=200,
    ),
    backbone=VocosBackboneArgs(
        input_channels=512,
        dim=768,
        intermediate_dim=2304,
        num_layers=12,
        adanorm_num_embeddings=4,
    ),
    head=ISTFTHeadArgs(
        dim=768,
        n_fft=1280,
        hop_length=320,
        padding="same",
    ),
)
