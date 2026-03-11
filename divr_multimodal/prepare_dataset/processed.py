from __future__ import annotations
from typing import List, Set
from dataclasses import dataclass
from divr_diagnosis import Diagnosis


@dataclass
class ProcessedText:
    text_key: str
    text: str

    @property
    def __dict__(self):
        return {
            "text_key": self.text_key,
            "text": self.text,
        }

    @staticmethod
    def from_json(json_data):
        return ProcessedText(**json_data)


@dataclass
class ProcessedSession:
    id: str
    speaker_id: str
    age: int | None
    gender: str
    diagnosis: List[Diagnosis]
    texts: List[ProcessedText]
    num_texts: int

    @property
    def __dict__(self):
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "age": self.age,
            "gender": self.gender,
            "diagnosis": [diagnosis.name for diagnosis in self.diagnosis],
            "texts": [text.__dict__ for text in self.texts],
            "num_texts": self.num_texts,
        }

    @property
    def best_diagnosis(self) -> Diagnosis:
        sorted_diagnosis = sorted(self.diagnosis, reverse=True)
        complete_diagnosis = list(
            filter(lambda x: not x.incompletely_classified, sorted_diagnosis)
        )
        if len(complete_diagnosis) > 0:
            return complete_diagnosis[0]
        return sorted_diagnosis[0]

    def diagnosis_names_at_level(self, level: int) -> Set[str]:
        diag_names = set()
        for diagnosis in self.diagnosis:
            diag_names.add(diagnosis.at_level(level).name)
        return diag_names

    def diagnosis_at_level(self, level: int) -> List[Diagnosis]:
        diags = {}
        for diagnosis in self.diagnosis:
            diag = diagnosis.at_level(level)
            if diag.name not in diags:
                diags[diag.name] = diag
        return sorted(list(diags.values()), reverse=True)


@dataclass
class ProcessedDataset:
    db_name: str
    train_sessions: List[ProcessedSession]
    val_sessions: List[ProcessedSession]
    test_sessions: List[ProcessedSession]

    @property
    def __dict__(self):
        return {
            "db_name": self.db_name,
            "train_sessions": [session.__dict__ for session in self.train_sessions],
            "val_sessions": [session.__dict__ for session in self.val_sessions],
            "test_sessions": [session.__dict__ for session in self.test_sessions],
        }

    @property
    def all_sessions(self) -> List[ProcessedSession]:
        return self.train_sessions + self.val_sessions + self.test_sessions
