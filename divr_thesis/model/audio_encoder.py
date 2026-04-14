from importlib import import_module

import torch
from s3prl.nn import S3PRLUpstream
from data_loader import InputTensors


HF_AUDIO_MODEL_ALIASES = {
    "hear": "google/hear",
    "clap": "laion/clap-htsat-fused",
}


class _AudioEncoderBase(torch.nn.Module):
    model_name: str
    device: torch.device

    def _prepare_batch(self, batch: InputTensors) -> InputTensors:
        audios, audio_lens = batch
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(
                audios,
                device=self.device,
                dtype=torch.float32,
            )
        elif audios.device != self.device:
            audios = audios.to(self.device)
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(
                audio_lens,
                device=self.device,
                dtype=torch.long,
            )
        elif audio_lens.device != self.device:
            audio_lens = audio_lens.to(self.device)
        return audios, audio_lens

    def _extract_feature_lens(
        self,
        feature: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if feature.dim() == 2:
            feature = feature.unsqueeze(1)
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.size(0) == feature.size(0)
            and attention_mask.size(1) == feature.size(1)
        ):
            return attention_mask.sum(dim=1).to(dtype=torch.long)
        return torch.full(
            (feature.size(0),),
            feature.size(1),
            dtype=torch.long,
            device=feature.device,
        )


class S3PrlFrozen(_AudioEncoderBase):
    model_name: str
    device: torch.device

    def __init__(
        self, model_name: str, device: torch.device
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = S3PRLUpstream(self.model_name).eval().to(self.device)

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        audios, audio_lens = self._prepare_batch(batch)
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        feature_lens = all_hs_len[0]
        return feature, feature_lens


class HuggingFaceFrozen(_AudioEncoderBase):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        try:
            transformers = import_module("transformers")
            AutoModel = transformers.AutoModel
            AutoProcessor = transformers.AutoProcessor
        except ModuleNotFoundError as error:
            raise ImportError(
                "HuggingFace audio models require transformers. "
                "Install it with: pip install transformers"
            ) from error

        resolved_name = HF_AUDIO_MODEL_ALIASES.get(
            model_name.lower(),
            model_name,
        )
        self.model_name = resolved_name
        self.device = device
        self.sample_rate = sample_rate
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).eval().to(self.device)

    def _processor_inputs(
        self,
        waveforms: list[torch.Tensor],
    ) -> dict[str, object]:
        for audio_key in ("audios", "audio", "raw_speech", "input_values"):
            try:
                inputs = self.processor(
                    **{audio_key: waveforms},
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                if isinstance(inputs, dict):
                    return {
                        key: value.to(self.device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in inputs.items()
                    }
            except TypeError:
                continue
        raise ValueError(
            f"Unsupported HuggingFace processor input for: {self.model_name}"
        )

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        audios, audio_lens = self._prepare_batch(batch)
        waveforms = [
            audios[index, : int(audio_lens[index].item())].detach().cpu()
            for index in range(audios.size(0))
        ]
        model_inputs = self._processor_inputs(waveforms)
        outputs = self.model(**model_inputs)

        attention_mask = model_inputs.get("attention_mask")
        if hasattr(outputs, "last_hidden_state") and isinstance(
            outputs.last_hidden_state,
            torch.Tensor,
        ):
            feature = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            feature = outputs.hidden_states[-1]
        elif hasattr(outputs, "audio_embeds") and isinstance(
            outputs.audio_embeds,
            torch.Tensor,
        ):
            feature = outputs.audio_embeds.unsqueeze(1)
        elif hasattr(outputs, "pooler_output") and isinstance(
            outputs.pooler_output,
            torch.Tensor,
        ):
            feature = outputs.pooler_output.unsqueeze(1)
        elif isinstance(outputs, (tuple, list)) and outputs and isinstance(
            outputs[0],
            torch.Tensor,
        ):
            feature = outputs[0]
        else:
            raise ValueError(
                "Cannot infer features from HuggingFace model output: "
                f"{self.model_name}"
            )

        if feature.dim() == 2:
            feature = feature.unsqueeze(1)
        feature_lens = self._extract_feature_lens(feature, attention_mask)
        return feature, feature_lens


class AudioEncoder(_AudioEncoderBase):
    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__()
        normalized_name = model_name.strip().lower()
        if normalized_name in HF_AUDIO_MODEL_ALIASES:
            self.encoder = HuggingFaceFrozen(
                model_name=model_name,
                device=device,
            )
        else:
            self.encoder = S3PrlFrozen(model_name=model_name, device=device)
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        return self.encoder(batch)
