from pathlib import Path
from typing import List

from class_argparse import ClassArgParser
from divr_diagnosis import diagnosis_maps

from ..benchmark import Benchmark
from ..task_generator import versions, generate_tasks
from ..statisticalanalysis.convert_csv import convert_text_csv


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="DiVR Multimodal")

    async def generate_text_tasks(
        self,
        version: str,
        data_store_path: Path,
        task_name: str = "",
        diagnosis_map: str = "USVAC_2025",
        diag_level: int = 0,
        datasets: List[str] = [],
        text_fields: List[str] = [],
        text_equals: List[str] = [],
    ) -> None:
        if not data_store_path.is_dir():
            raise ValueError(
                f"data_store_path does not exist: {data_store_path}"
            )
        if not hasattr(diagnosis_maps, diagnosis_map):
            raise ValueError(f"Unknown diagnosis map: {diagnosis_map}")
        if version not in versions:
            raise ValueError(
                f"Invalid version ({version}). Choose from {versions}"
            )
        if diag_level < 0:
            raise ValueError("diag_level must be >= 0")

        diag_map = getattr(diagnosis_maps, diagnosis_map)(allow_unmapped=False)
        selected_datasets = datasets if len(datasets) > 0 else None
        selected_text_fields = text_fields if len(text_fields) > 0 else None
        selected_text_equals = text_equals if len(text_equals) > 0 else None

        selected_task_name = (
            task_name.strip() if len(task_name.strip()) > 0 else None
        )

        await generate_tasks(
            version=version,
            source_path=data_store_path,
            diagnosis_map=diag_map,
            diag_level=diag_level,
            databases=selected_datasets,
            text_fields=selected_text_fields,
            text_equals=selected_text_equals,
            task_name=selected_task_name,
        )

        output_name = (
            selected_task_name
            if selected_task_name is not None
            else version
        )
        print(
            "Task generation complete. "
            f"Generated files are under divr_multimodal/tasks/{output_name}/"
        )

    async def inspect_task(
        self,
        storage_path: Path,
        task_path: Path,
        diagnosis_map: str = "USVAC_2025",
        diag_level: int = -1,
        quiet: bool = False,
    ) -> None:
        if not hasattr(diagnosis_maps, diagnosis_map):
            raise ValueError(f"Unknown diagnosis map: {diagnosis_map}")
        if not task_path.is_dir():
            raise ValueError(f"task_path is not a directory: {task_path}")
        selected_diag_level = None if diag_level < 0 else diag_level

        diag_map = getattr(diagnosis_maps, diagnosis_map)(allow_unmapped=False)
        benchmark = Benchmark(
            storage_path=storage_path,
            version="v1",
            quiet=quiet,
        )
        task = benchmark.load_task(
            task_path=task_path,
            diag_level=selected_diag_level,
            diagnosis_map=diag_map,
            load_texts=True,
        )

        print("Task loaded successfully")
        print(f"  max_diag_level: {task.max_diag_level}")
        print(f"  train size    : {len(task.train)}")
        print(f"  val size      : {len(task.val)}")
        print(f"  test size     : {len(task.test)}")
        print(f"  labels        : {task.unique_diagnosis(level=None)}")

        if len(task.train) > 0:
            sample = task.train[0]
            print("\nTrain sample")
            print(f"  id            : {sample.id}")
            print(f"  label         : {sample.label.name}")
            print(f"  num texts     : {len(sample.texts)}")
            preview = sample.texts[0][:120] if len(sample.texts) > 0 else ""
            print(f"  text preview  : {preview}")

    async def convert_text_csv(
        self,
        data_store_path: Path,
        csv_output: Path,
        diagnosis_map: str = "USVAC_2025",
        diag_level: int = 0,
        datasets: List[str] = [],
        text_fields: List[str] = [],
        text_equals: List[str] = [],
        labels: List[str] = [],
    ) -> None:
        if not data_store_path.is_dir():
            raise ValueError(
                f"data_store_path does not exist: {data_store_path}"
            )
        if not hasattr(diagnosis_maps, diagnosis_map):
            raise ValueError(f"Unknown diagnosis map: {diagnosis_map}")
        if diag_level < 0:
            raise ValueError("diag_level must be >= 0")

        diag_map = getattr(diagnosis_maps, diagnosis_map)(allow_unmapped=False)
        selected_datasets = datasets if len(datasets) > 0 else None
        selected_text_fields = text_fields if len(text_fields) > 0 else None
        selected_text_equals = text_equals if len(text_equals) > 0 else None
        selected_labels = labels if len(labels) > 0 else None

        await convert_text_csv(
            source_path=data_store_path,
            output_csv_path=csv_output,
            diagnosis_map=diag_map,
            diag_level=diag_level,
            databases=selected_datasets,
            text_fields=selected_text_fields,
            text_equals=selected_text_equals,
            labels=selected_labels,
        )


if __name__ == "__main__":
    Main()()
