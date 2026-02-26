from pathlib import Path
from typing import Callable, Dict, List, Set
from divr_diagnosis import Diagnosis, DiagnosisMap

from ...prepare_dataset.database_generator import DatabaseGenerator
from ...prepare_dataset.processed import (
    ProcessedText,
    ProcessedSession,
)
from ..task import Task

TextFilter = Callable[[List[ProcessedText]], List[ProcessedText]]


class Base:
    DB_NAME: str

    def __init__(
        self,
        source_path: Path,
    ) -> None:
        self.__source_path = source_path
        self.__db_source_path = Path(f"{source_path}/{self.DB_NAME}")

    async def init(
        self,
        diagnosis_map: DiagnosisMap,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ):
        database_generator = DatabaseGenerator(
            train_split=0.7,
            test_split=0.2,
            random_seed=42,
        )
        sessions = await self.prepare_dataset(
            source_path=self.__db_source_path,
            allow_incomplete_classification=allow_incomplete_classification,
            min_tasks=min_tasks,
            diagnosis_map=diagnosis_map,
        )
        self.dataset = database_generator.generate(
            db_name=self.DB_NAME,
            sessions=sessions,
        )

    async def collect_diagnosis_terms(self) -> Set[str]:
        return await self._collect_diagnosis_terms(source_path=self.__db_source_path)

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        raise NotImplementedError()

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        raise NotImplementedError()

    def to_text_key(self, source_text: ProcessedText) -> str:
        return source_text.key

    def all_train(self, level: int) -> List[Task]:
        return self.to_individual_text_tasks(
            self.dataset.train_sessions, level=level, text_filter=None
        )

    def all_val(self, level: int) -> List[Task]:
        return self.to_individual_text_tasks(
            self.dataset.val_sessions, level=level, text_filter=None
        )

    def all_test(self, level: int) -> List[Task]:
        return self.to_individual_text_tasks(
            self.dataset.test_sessions, level=level, text_filter=None
        )

    def all(self, level: int) -> List[Task]:
        return self.to_individual_text_tasks(
            sessions=self.dataset.all_sessions, level=level, text_filter=None
        )

    def count_per_diag(self, level: int) -> Dict[Diagnosis, set[str]]:
        counts: dict[Diagnosis, set[str]] = {}
        for session in self.dataset.all_sessions:
            root_diagnosis = session.best_diagnosis.at_level(level)
            if root_diagnosis not in counts:
                counts[root_diagnosis] = set()
            counts[root_diagnosis].add(session.speaker_id)
        return counts

    def to_individual_text_tasks(
        self,
        sessions: List[ProcessedSession],
        level: int,
        text_filter: TextFilter | None,
    ) -> List[Task]:
        tasks: List[Task] = []
        for session in sessions:
            root_diagnosis = session.best_diagnosis.at_level(level)
            if text_filter is None:
                texts = session.texts
            else:
                texts = text_filter(session.texts)
            for text_idx, text_item in enumerate(texts):
                task = Task(
                    id=f"{session.id}_{text_idx}",
                    speaker_id=session.speaker_id,
                    age=session.age,
                    gender=session.gender,
                    label=root_diagnosis,
                    text_keys=[self.to_text_key(text_item)],
                    texts=[text_item.text],
                )
                tasks.append(task)
        return tasks

    def to_multi_text_tasks(
        self,
        sessions: List[ProcessedSession],
        level: int,
        text_filter: TextFilter | None,
    ) -> List[Task]:
        tasks: List[Task] = []
        for session in sessions:
            root_diagnosis = session.best_diagnosis.at_level(level)
            if text_filter is None:
                texts = session.texts
            else:
                texts = text_filter(session.texts)
            task = Task(
                id=f"{session.id}",
                speaker_id=session.speaker_id,
                age=session.age,
                gender=session.gender,
                label=root_diagnosis,
                text_keys=list(map(self.to_text_key, texts)),
                texts=[item.text for item in texts],
            )
            tasks.append(task)
        return tasks
