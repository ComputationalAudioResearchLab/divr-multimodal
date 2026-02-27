from tqdm import tqdm
from pathlib import Path
from divr_benchmark import Benchmark
from divr_benchmark.task_generator import DatabaseFunc, Dataset
from divr_diagnosis import diagnosis_maps
from class_argparse import ClassArgParser


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="DiVR Tutorial")

    async def load_task(self):
        bench = Benchmark(
            storage_path="/home/storage",
            version="v1",
            quiet=False,
            sample_rate=16000,
        )
        diag_map = diagnosis_maps.USVAC_2025(allow_unmapped=False)
        # Load current task, this will download the data if it hasn't been downloaded already
        bench_path = Path(__file__).parent.parent.resolve()
        task = bench.load_task(
            task_path=Path(f"{bench_path}/divr_StatisticalAnalysis/tasks/femh_tasks"),
            diag_level=None,
            diagnosis_map=diag_map,
            load_audios=True,
        )
        # print some stats and loop over the datasets for an example
        print(dict(zip(task.unique_diagnosis(), task.train_class_weights())))
        # You can expand upon this function to create a training loop
        for point in tqdm(task.train, desc="training"):
            label = task.diag_to_index(point.label)
            audio = point.audio
        for point in tqdm(task.val, desc="validating"):
            label = task.diag_to_index(point.label)
            audio = point.audio
        for point in tqdm(task.test, desc="testing"):
            label = task.diag_to_index(point.label)
            audio = point.audio

    async def generate_task(self):
        # example of generating task, doesn't generate a task as of now
        # you'll have to fix the filter func for it to start working
        bench = Benchmark(
            storage_path="/home/storage",
            version="v1",
            quiet=False,
            sample_rate=16000,
        )
        diag_map = diagnosis_maps.USVAC_2025(allow_unmapped=False)
        curdir = Path(__file__).parent.resolve()
        pardir = curdir.parent.resolve()
        task_path = Path(f"{pardir}/divr_StatisticalAnalysis/tasks/femh_tasks")
        await bench.generate_task(
            filter_func=self.__filter_func,
            task_path=task_path,
            diagnosis_map=diag_map,
            allow_incomplete_classification=False,
        )

    async def __filter_func(self, database_func: DatabaseFunc):
        # You can filter the data by min_tasks, so thate every speaker has at least N audios
        # this is called 'task' because in most datasets the audios represent different vocal tasks
        db = await database_func(name="femh", min_tasks=None)
        diag_map = diagnosis_maps.USVAC_2025(allow_unmapped=False)
        diag_level = diag_map.max_diag_level

        def filter_unclassified(tasks):
            # example of filtering tasks by label
            # You can also get task.speaker_id which can be used to count
            # number of diag/speaker and restrict which diags are used for the dataset
            return [task for task in tasks if not task.label.incompletely_classified]

        return Dataset(
            train=filter_unclassified(db.all_train(level=diag_level)),
            val=filter_unclassified(db.all_val(level=diag_level)),
            test=filter_unclassified(db.all_test(level=diag_level)),
        )


if __name__ == "__main__":
    Main()()
