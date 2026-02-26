from typing import List
from dataclasses import dataclass
from divr_diagnosis import Diagnosis


@dataclass
class Task:
    id: str
    speaker_id: str
    age: int | None
    gender: str
    label: Diagnosis
    text_keys: List[str]
    texts: List[str]
