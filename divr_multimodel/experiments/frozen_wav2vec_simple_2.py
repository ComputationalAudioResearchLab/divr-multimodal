import torch
import torch.optim
from torch import nn
from pathlib import Path
from .base.trainer import Trainer as BaseTrainer
from ..model import S3PrlFrozen, Simple
from .base.hparams import HParams
from ..data_loader import DataLoaderWithFeature
from .database import tasks, TASK_KEYS


class Trainer(BaseTrainer):

    def __init__(self, task_key: TASK_KEYS) -> None:
        device = torch.device("cuda")
        model_name = "Frozen_Wav2Vec_Simple_2"
        feature = S3PrlFrozen(
            model_name="wav2vec_large",
            device=device,
            checkpoint_path=Path(
                f"{HParams.cache_path}/checkpoints/wav2vec_large/{task_key}"
            ),
        )
        data_loader = DataLoaderWithFeature(
            data_root=Path(f"{HParams.cache_path}/data"),
            sample_rate=HParams.sample_rate,
            feature=feature,
            device=device,
            batch_size=32,
            random_seed=HParams.random_seed,
            shuffle_train=True,
            database=tasks[task_key],
        )
        num_classes = len(data_loader.unique_diagnosis)
        class_weights = data_loader.class_counts.sum() / data_loader.class_counts
        model = Simple(
            input_size=6656,
            num_classes=num_classes,
            checkpoint_path=Path(
                f"{HParams.cache_path}/checkpoints/{model_name}/{task_key}"
            ),
        )
        model.to(device=device)
        num_epochs = 202
        hparams = HParams(
            criterion=nn.CrossEntropyLoss(weight=class_weights),
            data_loader=data_loader,
            device=device,
            feature=feature,
            model=model,
            model_name=model_name,
            task_key=task_key,
            num_classes=num_classes,
            num_epochs=num_epochs,
            optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-6),
            save_epochs=list(range(0, num_epochs + 1, num_epochs // 10)),
            confusion_epochs=list(range(0, num_epochs + 1, 10)),
        )
        super().__init__(hparams=hparams)
        self.run()
