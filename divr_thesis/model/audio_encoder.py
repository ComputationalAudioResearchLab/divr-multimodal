from importlib import import_module

import numpy as np
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


class _HuggingFaceFrozenBase(_AudioEncoderBase):
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

    def _feature_from_outputs(
        self,
        outputs: object,
    ) -> torch.Tensor:
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
        return feature

    def _processor_inputs(
        self,
        waveforms: list[np.ndarray],
    ) -> dict[str, object]:
        for audio_key in ("audio", "audios", "raw_speech", "input_values"):
            try:
                inputs = self.processor(
                    **{audio_key: waveforms},
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                if hasattr(inputs, "items"):
                    return {
                        key: value.to(self.device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in inputs.items()
                    }
            except (TypeError, ValueError):
                continue
        raise ValueError(
            f"Unsupported HuggingFace processor input for: {self.model_name}"
        )

    def _waveforms_from_batch(self, batch: InputTensors) -> list[np.ndarray]:
        audios, audio_lens = self._prepare_batch(batch)
        return [
            audios[index, : int(audio_lens[index].item())]
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
            for index in range(audios.size(0))
        ]

    def _forward_waveforms(
        self,
        waveforms: list[np.ndarray],
    ) -> InputTensors:
        model_inputs = self._processor_inputs(waveforms)
        outputs = self.model(**model_inputs)
        attention_mask = model_inputs.get("attention_mask")
        feature = self._feature_from_outputs(outputs)
        feature_lens = self._extract_feature_lens(feature, attention_mask)
        return feature, feature_lens

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        waveforms = self._waveforms_from_batch(batch)
        return self._forward_waveforms(waveforms)


class HuggingFaceClapFrozen(_HuggingFaceFrozenBase):
    def _pool_hidden_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() <= 2:
            return state
        reduce_dims = tuple(range(1, state.dim() - 1))
        return state.mean(dim=reduce_dims)

    def _forward_waveforms(
        self,
        waveforms: list[np.ndarray],
    ) -> InputTensors:
        model_inputs = self._processor_inputs(waveforms)

        # Use CLAP audio backbone hidden states and concatenate all layers.
        if (
            "input_features" in model_inputs
            and hasattr(self.model, "audio_model")
        ):
            audio_kwargs: dict[str, object] = {
                "input_features": model_inputs["input_features"],
                "output_hidden_states": True,
                "return_dict": True,
            }
            if "is_longer" in model_inputs:
                audio_kwargs["is_longer"] = model_inputs["is_longer"]

            audio_outputs = self.model.audio_model(**audio_kwargs)
            if (
                hasattr(audio_outputs, "hidden_states")
                and audio_outputs.hidden_states
            ):
                hidden_states = [
                    state
                    for state in audio_outputs.hidden_states
                    if isinstance(state, torch.Tensor)
                ]
                pooled_states = [
                    self._pool_hidden_state(state)
                    for state in hidden_states
                ]
                feature = torch.cat(pooled_states, dim=-1).unsqueeze(1)
            else:
                feature = self._feature_from_outputs(audio_outputs)

            feature_lens = torch.ones(
                (feature.size(0),),
                dtype=torch.long,
                device=feature.device,
            )
            return feature, feature_lens

        outputs = self.model(**model_inputs)
        attention_mask = model_inputs.get("attention_mask")
        feature = self._feature_from_outputs(outputs)
        feature_lens = self._extract_feature_lens(feature, attention_mask)
        return feature, feature_lens


class HuggingFaceHearFrozen(_AudioEncoderBase):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        resolved_name = HF_AUDIO_MODEL_ALIASES.get(
            model_name.lower(),
            model_name,
        )
        self.model_name = resolved_name
        self.device = device
        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            raise ValueError(
                "HeAR requires 16kHz input. "
                f"Received sample_rate={self.sample_rate}."
            )

        try:
            tf = import_module("tensorflow")
            hub = import_module("huggingface_hub")
            snapshot_download = hub.snapshot_download
        except ModuleNotFoundError as error:
            raise ImportError(
                "HeAR encoder requires tensorflow and huggingface_hub. "
                "Install with: pip install tensorflow huggingface_hub"
            ) from error

        self._tf = tf
        tf_gpus = self._tf.config.list_physical_devices("GPU")
        if self.device.type == "cuda" and tf_gpus:
            self._tf_infer_device = "/GPU:0"
        else:
            self._tf_infer_device = "/CPU:0"
        self._fallback_logged = False
        try:
            model_dir = snapshot_download(
                repo_id=self.model_name,
                allow_patterns=[
                    "saved_model.pb",
                    "variables/*",
                    "fingerprint.pb",
                ],
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to download HeAR SavedModel from Hugging Face. "
                "Check your model access and authentication for "
                f"{self.model_name}."
            ) from error

        self._model = self._tf.saved_model.load(model_dir)
        signatures = getattr(self._model, "signatures", {})
        self._serving = signatures.get("serving_default")
        if self._serving is None:
            raise ValueError(
                "HeAR model is missing serving_default signature: "
                f"{self.model_name}"
            )

    def _split_hear_waveform(self, waveform: np.ndarray) -> list[np.ndarray]:
        segment_len = max(1, int(self.sample_rate * 2))
        if waveform.size == 0:
            return [np.zeros((segment_len,), dtype=np.float32)]

        segments: list[np.ndarray] = []
        for start in range(0, waveform.shape[0], segment_len):
            segment = waveform[start:start + segment_len]
            if segment.shape[0] < segment_len:
                segment = np.pad(
                    segment,
                    (0, segment_len - segment.shape[0]),
                    mode="constant",
                )
            segments.append(segment.astype(np.float32, copy=False))
        return segments

    def _infer_hear_embeddings(
        self,
        batch_waveforms: np.ndarray,
    ) -> np.ndarray:
        def _run(device_name: str):
            with self._tf.device(device_name):
                tf_input = self._tf.convert_to_tensor(
                    batch_waveforms,
                    dtype=self._tf.float32,
                )
                try:
                    return self._serving(x=tf_input)
                except TypeError:
                    return self._serving(tf_input)

        try:
            outputs = _run(self._tf_infer_device)
        except Exception as error:
            if self._tf_infer_device == "/GPU:0":
                self._tf_infer_device = "/CPU:0"
                if not self._fallback_logged:
                    print(
                        "[HeAR] GPU inference failed; switching to CPU "
                        f"for stability. Reason: {type(error).__name__}"
                    )
                    self._fallback_logged = True
                outputs = _run(self._tf_infer_device)
            else:
                raise

        if isinstance(outputs, dict):
            if "output_0" in outputs:
                output_tensor = outputs["output_0"]
            else:
                output_tensor = next(iter(outputs.values()))
        else:
            output_tensor = outputs

        if hasattr(output_tensor, "numpy"):
            embeddings = output_tensor.numpy()
        else:
            embeddings = np.asarray(output_tensor)

        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        elif embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings.astype(np.float32, copy=False)

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        audios, audio_lens = self._prepare_batch(batch)
        sample_features: list[torch.Tensor] = []
        sample_lens: list[int] = []

        for index in range(audios.size(0)):
            waveform = (
                audios[index, : int(audio_lens[index].item())]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            segments = self._split_hear_waveform(waveform)
            segment_batch = np.stack(segments, axis=0)
            segment_embeddings = self._infer_hear_embeddings(segment_batch)
            segment_feature = torch.tensor(
                segment_embeddings,
                dtype=torch.float32,
                device=self.device,
            )
            sample_features.append(segment_feature)
            sample_lens.append(segment_feature.size(0))

        max_len = max(feature.size(0) for feature in sample_features)
        feature_size = sample_features[0].size(1)
        feature = torch.zeros(
            (len(sample_features), max_len, feature_size),
            dtype=sample_features[0].dtype,
            device=sample_features[0].device,
        )
        for index, per_sample in enumerate(sample_features):
            current_len = per_sample.size(0)
            feature[index, :current_len] = per_sample

        feature_lens = torch.tensor(
            sample_lens,
            dtype=torch.long,
            device=feature.device,
        )
        return feature, feature_lens


class HuggingFaceGenericFrozen(_HuggingFaceFrozenBase):
    pass


class AudioEncoder(_AudioEncoderBase):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        normalized_name = model_name.strip().lower()
        resolved_name = HF_AUDIO_MODEL_ALIASES.get(
            normalized_name,
            model_name,
        )
        if normalized_name in HF_AUDIO_MODEL_ALIASES:
            resolved_normalized_name = resolved_name.strip().lower()
            if "clap" in resolved_normalized_name:
                self.encoder = HuggingFaceClapFrozen(
                    model_name=model_name,
                    device=device,
                    sample_rate=sample_rate,
                )
            elif resolved_normalized_name == "google/hear":
                self.encoder = HuggingFaceHearFrozen(
                    model_name=model_name,
                    device=device,
                    sample_rate=sample_rate,
                )
            else:
                self.encoder = HuggingFaceGenericFrozen(
                    model_name=model_name,
                    device=device,
                    sample_rate=sample_rate,
                )
        else:
            self.encoder = S3PrlFrozen(model_name=model_name, device=device)
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        return self.encoder(batch)
