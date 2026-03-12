from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


AudioBatch = List[np.ndarray]
"""
AudioBatch = collection of audios in a given batch
[
    audio_1,
    audio_2,
    audio_3,
    ... for batch
]
"""

InputArrays = Tuple[np.ndarray, np.ndarray]
"""
InputArrays[0]
    - contains audio data
    - shape = [B, S]
---
InputArrays[1]
    - contains length of each audio in batch
    - shape = [B]

---
- B = Batch size
- S = Sequence length
---
"""

InputTensors = Tuple[torch.Tensor, torch.Tensor]
"""
InputTensors[0]
    - contains feature data
    - type = torch.FloatTensor
    - shape = [B, S, H]
---
InputTensors[1]
    - contains length of each audio in batch
    - type = torch.LongTensor
    - shape = [B]

---
- B = Batch size
- S = Sequence length
- H = Feature length
---
"""

DemographicInfo = Tuple[torch.LongTensor, List[str]]
"""
Demographic information batch:
    1. ages: Tensor of shape [B] with age values or -1 for unknown
    2. genders: List of gender strings with length B
"""

DataPoint = Tuple[InputTensors, torch.LongTensor]
"""
Single data point consisting of input tensors and labels
"""

DataPointWithDemo = Tuple[InputTensors, torch.LongTensor, DemographicInfo]
"""
Data point with demographic information:
    1. Input tensors (audio features and lengths)
    2. Labels
    3. Demographic info (ages and genders)
"""

Dataset = List[Tuple[int | None, str, str, str]]
"""
Dataset defined as list of rows where each row has the structure:
    1. age = None or int
    2. gender = str
    3. label = str (pathology or normal)
    4. audio file path
"""

Database = Tuple[Dataset, Dataset, Dataset]
"""
Database consisting of training, evaluation and testing dataset
"""

TextTensors = Tuple[torch.Tensor, torch.Tensor]
"""
TextTensors[0]
    - contains token ids
    - type = torch.LongTensor
    - shape = [B, T]
---
TextTensors[1]
    - contains number of valid tokens for each sample
    - type = torch.LongTensor
    - shape = [B]
---
- B = Batch size
- T = Token sequence length
---
"""

BatchMetadata = Dict[str, List[Any]]


@dataclass(slots=True)
class TaskRecord:
    sample_id: str
    label: str
    audio_paths: List[str]
    texts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Batch:
    sample_ids: List[str]
    labels: torch.LongTensor
    audio_inputs: InputTensors | None
    text_inputs: TextTensors | None
    audio_paths: List[List[str]]
    selected_texts: List[str]
    metadata: BatchMetadata
