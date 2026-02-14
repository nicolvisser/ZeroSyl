import torch
from transformers import OPTConfig, OPTForCausalLM


class LanguageModel(OPTForCausalLM):
    def __init__(self, config: OPTConfig):
        super().__init__(config)

    def tokenize(self, units: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [
                torch.tensor(
                    [self.config.bos_token_id], dtype=torch.long, device=units.device
                ),
                units.long(),
            ]
        )
        return tokens

    def loglikelihoods(
        self, ids_list: list[torch.Tensor], normalize=True
    ) -> list[float]:
        for ids in ids_list:
            assert ids.ndim == 1, "Each input should be a 1D tensor of token IDs."
        seqlens = [len(ids) for ids in ids_list]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            ids_list, batch_first=True, padding_value=self.config.pad_token_id
        ).to(self.device)
        with torch.no_grad():
            logits = self(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        reduce_fn = torch.mean if normalize else torch.sum
        scores = [
            reduce_fn(tlp[:sl]).item() for tlp, sl in zip(token_log_probs, seqlens)
        ]
        return scores

    @classmethod
    def from_pretrained_file(cls, checkpoint_path: str) -> "LanguageModel":
        checkpoint = torch.load(checkpoint_path)
        cfg = OPTConfig(**checkpoint["cfg"])
        model = cls(cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        num_params = sum(map(torch.numel, model.parameters()))
        print(f"Model loaded with {num_params:,} parameters.")
        return model

    @classmethod
    def from_remote(
        cls,
        checkpoint_url: str = "https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-60kh-ZeroSylCollapsed-v040-k-9116.pt",
    ) -> "LanguageModel":
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url)
        cfg = OPTConfig(**checkpoint["cfg"])
        model = cls(cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        num_params = sum(map(torch.numel, model.parameters()))
        print(f"Model loaded with {num_params:,} parameters.")
        return model
