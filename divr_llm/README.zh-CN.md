# divr_llm

这个项目是自包含的，只使用 `divr_llm` 文件夹内的代码。

## Qwen2Audio LoRA

默认微调配置在 `lora/qwen2audio.yaml`，当前对接的是 `tasks/voiced_0_test` 任务集。

先查看解析后的配置，不启动训练：

```bash
python -m src --config lora/qwen2audio.yaml
```

列出可选的任务预设：

```bash
python -m src --list-tasks
```

通过 CLI 选择任务预设：

```bash
python -m src --config lora/qwen2audio.yaml --task voiced_1_test
```

生成测试集报告，包含混淆矩阵和准确率图：

```bash
python -m src --config lora/qwen2audio.yaml --task voiced_1_test --test
```

开始 LoRA 微调：

```bash
python -m src --config lora/qwen2audio.yaml --train
```

也可以训练和测试报告一起执行：

```bash
python -m src --config lora/qwen2audio.yaml --task voiced_1_test --train --test
```

`--task` 会切换 `train/val/test` 的任务 YAML。如果所选任务的输出格式不同，还需要同步调整 YAML 配置里的 `data.prompt_template` 和 `data.response_key`。

## 代码结构

- `src/__main__.py`：只保留 CLI。
- `data_loader/qwen2audio.py`：任务 YAML 解析与音频加载。
- `model/qwen2audio_lora.py`：Qwen2Audio LoRA 训练逻辑。
- `analysis/`：保存测试集混淆矩阵、准确率表和图表。

## 任务格式

每个任务 YAML 条目继续沿用当前格式：

- `text_keys`：本地或绝对音频路径
- `texts`：监督答案或目标文本

## 本地产物

模型缓存和训练输出都会写入 `model/`，这样项目可以保持独立，不依赖其他工作区。

测试集报告会写到 `analysis/qwen2audio_lora/<task_name>/` 下。

英文版说明见 `README.md`。
