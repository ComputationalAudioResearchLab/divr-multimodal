import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")


class MockBoard:
    def add_figure(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass


class TBoard:
    def __init__(self, tensorboard_path: Path):
        self.logpath = tensorboard_path / time.strftime("%Y_%b_%d-%H_%M_%S-%z")
        self.logpath.mkdir(exist_ok=True, parents=True)
        self.writer = self._build_writer()

    def _build_writer(self):
        try:
            from torch.utils.tensorboard.writer import SummaryWriter
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "tensorboard is not installed. Install tensorboard or use "
                "--disable-tensorboard."
            ) from error
        return SummaryWriter(self.logpath)

    def add_figure(self, *args, **kwargs):
        return self.writer.add_figure(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        return self.writer.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        return self.writer.add_scalars(*args, **kwargs)

    def flush(self):
        return self.writer.flush()

    def close(self):
        return self.writer.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)

    def launch(self):
        try:
            from tensorboard.program import TensorBoard
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "tensorboard is not installed"
            ) from error
        tb = TensorBoard()
        tb.configure(argv=[None, "--logdir", str(self.logpath)])
        print("TensorBoard available at:: ", tb.launch())

    def keep_alive(self):
        try:
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            print("Exiting")
