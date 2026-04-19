# divr-multimodal

本项目用于探索嗓音障碍数据集中的文本信息，并基于 `divr-diagnosis` 提供可复现的文本任务生成与加载流程。

## 快速开始

生成 `femh` 文本任务：

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name femh \
    --diagnosis_map USVAC_2025 \
    --datasets femh \
    --text_fields age gender
```

检查已生成的任务：

```sh
python -m divr_multimodal.src inspect_task \
    /home/storage/data \
    /home/workspace/divr_multimodal/tasks/femh \
    --diagnosis_map USVAC_2025
```

## 安装

```sh
pip install divr-diagnosis
```

本仓库本地开发安装：

```sh
cd /home/workspace
pip install -e .
```

查看 CLI 帮助：

```sh
python -m divr_multimodal.src --help
```

## generate_text_tasks 参数说明

命令格式：

```sh
python -m divr_multimodal.src generate_text_tasks <version> <data_store_path> [options]
```

必填位置参数：

- `version`：任务生成版本。目前可用值为 `v1`。
- `data_store_path`：数据根目录路径，必须是已存在目录。

可选参数：

- `--task_name <name>`
  - 自定义输出任务目录名，输出到 `divr_multimodal/tasks/<name>/`。
  - 不传时默认使用 `<version>` 作为输出目录名。
- `--diagnosis_map <map_name>`
  - 诊断映射名称（来自 `divr_diagnosis.diagnosis_maps`）。
  - 默认：`USVAC_2025`。
- `--diag_level <int>`
  - 诊断层级，必须 `>= 0`。
  - 默认：`0`。
- `--datasets <dataset...>`
  - 选择一个或多个数据集。
  - 不传时使用该版本支持的全部数据集。
- `--text_fields <field...>`
  - 仅保留指定文本字段。
  - 不传时保留所有支持字段。
  - 传 `all` 等价于保留所有字段。
- `--text_equals <expr...>`
  - 按字段精确筛选样本。
  - 支持两种写法：
    - `key=value`（全局筛选）
    - `dataset.key=value`（按数据集作用域筛选，例如 `svd.svd_utterance=a_n`）
  - 可重复传入多个 `--text_equals`，也可在单次参数中用逗号分隔多个条件。
- `--labels <label...>`
  - 仅保留指定标签。

常见元数据字段：

- 通用字段：`dataset`、`speaker_id`、`age`、`gender`、`original_label`、`label`
- 数据集私有字段：
  - `femh`：`smoking`、`drinking`
  - `svd`：`svd_utterance`
  - `voiced`：`smoker`

## 常用示例

按标签过滤：

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name organic_only \
    --diagnosis_map USVAC_2025 \
    --datasets femh svd \
    --labels organic \
    --text_fields dataset age gender label
```

按层级生成标签：

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name femh_level1 \
    --diagnosis_map USVAC_2025 \
    --diag_level 1 \
    --datasets femh
```

多条件元数据过滤：

```sh
python -m divr_multimodal.src generate_text_tasks \
    v1 /home/storage/data \
    --task_name female_organic \
    --diagnosis_map USVAC_2025 \
    --datasets femh \
    --text_fields age gender smoking label \
    --text_equals gender=female \
    --text_equals label=organic
```

## Python API 用法

```python
import asyncio
from pathlib import Path

from divr_diagnosis import diagnosis_maps
from divr_multimodal import Benchmark
from divr_multimodal.task_generator import DatabaseFunc, Dataset


async def main():
    benchmark = Benchmark(
        storage_path="/home/storage",
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
        text_fields=["gender", "smoking"],
    )


asyncio.run(main())
```
