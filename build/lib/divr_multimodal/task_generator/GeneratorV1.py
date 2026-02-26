from pathlib import Path
from divr_diagnosis import DiagnosisMap
from typing import Awaitable, Callable, Dict, List

from .generator import Generator, DatabaseFunc, Dataset
from .databases import FEMH, SVD, Base as Database


class GeneratorV1(Generator):

    __db_map = {
        FEMH.DB_NAME: FEMH,
        SVD.DB_NAME: SVD,
    }

    async def collect_diagnosis_terms(
        self, source_path: Path
    ) -> Dict[str, List[str]]:
        dbs = [FEMH, SVD]
        terms = {}
        for db in dbs:
            db_instance = db(source_path=source_path)
            db_terms = await db_instance.collect_diagnosis_terms()
            for term in db_terms:
                if term not in terms:
                    terms[term] = set()
                terms[term].add(db.DB_NAME)
        terms.pop("", None)
        return terms

    async def count_for_diag_map(
        self, db_name: str, source_path: Path, diag_map: DiagnosisMap
    ) -> Database:
        db = self.__db_map[db_name](source_path=source_path)
        await db.init(
            diagnosis_map=diag_map,
            allow_incomplete_classification=True,
            min_tasks=None,
        )
        return db

    async def generate_task(
        self,
        source_path: Path,
        filter_func: Callable[[DatabaseFunc], Awaitable[Dataset]],
        task_path: Path,
        diagnosis_map: DiagnosisMap,
        allow_incomplete_classification: bool = False,
        text_fields: List[str] | None = None,
    ) -> None:
        async def __database(
            name: str,
            min_tasks: int | None = None,
        ) -> Database:
            db_name = name.lower()
            if db_name not in self.__db_map:
                supported = list(self.__db_map.keys())
                raise ValueError(
                    f"Unsupported database: {name}. "
                    f"Supported databases: {supported}"
                )
            db = self.__db_map[db_name](source_path=source_path)
            await db.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=(
                    allow_incomplete_classification
                ),
                min_tasks=min_tasks,
            )
            return db

        tasks = await filter_func(__database)
        normalized_text_fields = self.normalize_text_fields(text_fields)
        task_path.mkdir(parents=True, exist_ok=True)
        self.to_task_file(
            tasks=tasks.train,
            output_path=Path(f"{task_path}/train"),
            text_fields=normalized_text_fields,
        )
        self.to_task_file(
            tasks=tasks.val,
            output_path=Path(f"{task_path}/val"),
            text_fields=normalized_text_fields,
        )
        self.to_task_file(
            tasks=tasks.test,
            output_path=Path(f"{task_path}/test"),
            text_fields=normalized_text_fields,
        )

    async def __call__(
        self,
        source_path: Path,
        tasks_path: Path,
        diagnosis_map: DiagnosisMap,
        databases: List[str] | None = None,
        text_fields: List[str] | None = None,
    ) -> None:
        print("Generating text tasks for benchmark v1")
        tasks_path.mkdir(parents=True, exist_ok=True)
        normalized_text_fields = self.normalize_text_fields(text_fields)

        if databases is None:
            selected_databases = list(self.__db_map.keys())
        else:
            selected_databases = [
                name.strip().lower() for name in databases if name.strip()
            ]
            if len(selected_databases) == 0:
                raise ValueError("databases cannot be empty")

        unsupported = [
            db_name
            for db_name in selected_databases
            if db_name not in self.__db_map
        ]
        if len(unsupported) > 0:
            raise ValueError(
                f"Unsupported database(s): {unsupported}. "
                f"Supported: {list(self.__db_map.keys())}"
            )

        async def _init_db(db_cls):
            db = db_cls(source_path=source_path)
            await db.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=False,
                min_tasks=None,
            )
            return db

        train_tasks = []
        val_tasks = []
        test_tasks = []

        for db_name in selected_databases:
            db = await _init_db(self.__db_map[db_name])
            train_tasks.extend(db.all_train(level=0))
            val_tasks.extend(db.all_val(level=0))
            test_tasks.extend(db.all_test(level=0))

        self.to_task_file(
            tasks=train_tasks,
            output_path=Path(f"{tasks_path}/train"),
            text_fields=normalized_text_fields,
        )
        self.to_task_file(
            tasks=val_tasks,
            output_path=Path(f"{tasks_path}/val"),
            text_fields=normalized_text_fields,
        )
        self.to_task_file(
            tasks=test_tasks,
            output_path=Path(f"{tasks_path}/test"),
            text_fields=normalized_text_fields,
        )
