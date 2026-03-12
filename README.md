# divr-multimodal
The purpose of this experiment is to explore the textual information in the voice disorder dataset.

# The latest version of divr-benchmark has not been released yet. Please install it using femh as follows.
export RELEASE_VERSION=0.1.4
pip install git+https://github.com/ComputationalAudioResearchLab/divr-benchmark.git

# DiVR Multimodal (Text Task) - Benchmark

This repository provides a text-focused benchmark workflow for disordered voice datasets, built on top of the `divr-diagnosis` label standardization toolkit.

## Quick Start

Generate `femh` text tasks:

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name femh \
    --diagnosis_map USVAC_2025 \
    --datasets femh \
    --text_fields age gender
```

Load and inspect the generated task:

```sh
python -m divr_multimodal.src inspect_task \
    /home/storage/data \
    /home/workspace/divr_multimodal/tasks/femh \
    --diagnosis_map USVAC_2025
```

## Installation

```sh
pip install divr-diagnosis
```

Install this library (editable, for local development):

```sh
cd /home/workspace
pip install -e .
```

If you are using this repository directly (without publishing package install), run from workspace root so Python can import `divr_multimodal`.

You can then run the package entrypoint:

```sh
python -m divr_multimodal.src --help
```

### `__main__.py` CLI examples

Generate task files with user-selected datasets:

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name femh \
    --diagnosis_map USVAC_2025 \
    --datasets femh \
    --text_fields age gender
```

Inspect an existing task:

```sh
python -m divr_multimodal.src inspect_task \
    /home/storage/data \
    /home/workspace/divr_multimodal/tasks/femh \
    --diagnosis_map USVAC_2025
```

Filter by exact text metadata value (example: only `svd_utterance=a_n`):

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name svd_a_n \
    --diagnosis_map USVAC_2025 \
    --datasets svd \
    --text_fields age gender svd_utterance \
    --text_equals svd_utterance=a_n
```

Filter generated tasks by label at the same time:

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name organic_only \
    --diagnosis_map USVAC_2025 \
    --datasets femh svd \
    --labels organic \
    --text_fields dataset age gender label
```

When multiple datasets are selected, scope a filter to one dataset:

```sh
python -m divr_multimodal.src convert_text_csv \
    /home/storage/data \
    /home/workspace/texts.csv \
    --diagnosis_map USVAC_2025 \
    --diag_level 1 \
    --datasets femh svd \
    --text_fields dataset age gender label \
    --labels organic \
    --text_equals svd.svd_utterance=a_n
```

`dataset.key=value` only applies to that dataset (e.g. `svd.svd_utterance=a_n`), while plain `key=value` remains a global filter.

Export CSV with the same metadata filter:

```sh
python -m divr_multimodal.src convert_text_csv \
    /home/storage/data \
    /home/workspace/tmp/svd_a_n.csv \
    --diagnosis_map USVAC_2025 \
    --datasets svd \
    --text_fields svd_utterance label \
    --text_equals svd_utterance=a_n
```

## How to use

This library supports:
- generating text tasks from raw dataset storage
- loading existing task YAML files
- mapping diagnosis labels to class indices for training/evaluation

Current `v1` task generation supports databases:
- `femh`
- `svd`

You can choose which databases to generate from by passing `databases=[...]`
to the task generation API.

The benchmark expects datasets under:
- `<storage_path>/data`

For example, if your data is in `/home/storage/data`, set `storage_path="/home/storage"`.

### Generating tasks

```python
import asyncio
from pathlib import Path

from divr_diagnosis import diagnosis_maps
from divr_multimodal import Benchmark
from divr_multimodal.task_generator import DatabaseFunc, Dataset


async def main():
    benchmark = Benchmark(
        storage_path="/home/storage",  # expects /home/storage/data
        version="v1",
        quiet=False,
    )
    diag_map = diagnosis_maps.USVAC_2025(allow_unmapped=False)

    async def filter_func(database_func: DatabaseFunc):
        db = await database_func(name="femh", min_tasks=None)
        diag_level = diag_map.max_diag_level

        def filter_unclassified(tasks):
            return [task for task in tasks if not task.label.incompletely_classified]

        return Dataset(
            train=filter_unclassified(db.all_train(level=diag_level)),
            val=filter_unclassified(db.all_val(level=diag_level)),
            test=filter_unclassified(db.all_test(level=diag_level)),
        )

    await benchmark.generate_task(
        filter_func=filter_func,
        task_path=Path("/home/workspace/tmp/tasks/femh_text"),
        diagnosis_map=diag_map,
        allow_incomplete_classification=False,
        text_fields=["gender", "smoking"],  # e.g. choose only selected fields
    )


asyncio.run(main())
```

### Using existing tasks

Most task APIs accept `level` (diagnosis level). When `level=None`, the maximum diagnosis level in the loaded task is used.

```python
from pathlib import Path

from divr_diagnosis import diagnosis_maps
from divr_multimodal import Benchmark

benchmark = Benchmark(
    storage_path="/home/storage",
    version="v1",
    quiet=False,
)
diag_map = diagnosis_maps.USVAC_2025(allow_unmapped=False)

task = benchmark.load_task(
    task_path=Path("/home/workspace/tmp/tasks/femh_text"),
    diag_level=None,
    diagnosis_map=diag_map,
    load_texts=True,
)

# Training split
for train_point in task.train:
    point_id = train_point.id
    texts = train_point.texts
    label = task.diag_to_index(diag=train_point.label, level=None)

# Validation split
for val_point in task.val:
    point_id = val_point.id
    texts = val_point.texts
    label = task.diag_to_index(diag=val_point.label, level=None)

# Test split
for test_point in task.test:
    point_id = test_point.id
    texts = test_point.texts
    label = task.diag_to_index(diag=test_point.label, level=None)

# Class weights (for weighted CE loss)
class_weights = task.train_class_weights(level=None)

# Convert predicted index back to diagnosis
diagnosis = task.index_to_diag(index=0, level=None)
print(diagnosis.name)

# Get all diagnosis names present
diagnosis_names = task.unique_diagnosis(level=None)
print(diagnosis_names)
```