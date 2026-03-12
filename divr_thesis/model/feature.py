import torch
from s3prl.nn import S3PRLUpstream
from data_loader import InputTensors


class S3PrlFrozen(torch.nn.Module):
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
        batch_inputs, batch_lens = batch
        audios = batch_inputs
        audio_lens = batch_lens
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
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        feature_lens = all_hs_len[0]
        return feature, feature_lens
