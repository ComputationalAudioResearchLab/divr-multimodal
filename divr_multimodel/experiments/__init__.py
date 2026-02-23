from .frozen_data2vec_simple import Trainer as Frozen_Data2Vec_Simple
from .frozen_data2vec_simple_2 import Trainer as Frozen_Data2Vec_Simple_2
from .frozen_unispeechSAT_simple import Trainer as Frozen_unispeechSAT_Simple
from .frozen_unispeechSAT_simple_2 import Trainer as Frozen_unispeechSAT_Simple_2
from .frozen_wav2vec_simple import Trainer as Frozen_Wav2Vec_Simple
from .frozen_wav2vec_simple_2 import Trainer as Frozen_Wav2Vec_Simple_2
from typing import Literal
from .database import TASK_KEYS

experiments = {
    "frozen_data2vec_simple": Frozen_Data2Vec_Simple,
    "frozen_unispeechSAT_simple": Frozen_unispeechSAT_Simple,
    "frozen_wav2vec_simple": Frozen_Wav2Vec_Simple,
    "frozen_data2vec_simple_2": Frozen_Data2Vec_Simple_2,
    "frozen_unispeechSAT_simple_2": Frozen_unispeechSAT_Simple_2,
    "frozen_wav2vec_simple_2": Frozen_Wav2Vec_Simple_2,
}
EXPERIMENT_KEYS = Literal[
    "frozen_data2vec_simple",
    "frozen_unispeechSAT_simple",
    "frozen_wav2vec_simple",
    "frozen_data2vec_simple_2",
    "frozen_unispeechSAT_simple_2",
    "frozen_wav2vec_simple_2",
]

__all__ = ["TASK_KEYS"]
