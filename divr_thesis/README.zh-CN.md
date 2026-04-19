# DIVR Thesis（中文说明）

本仓库提供统一入口，可直接从 `src/__main__.py` 运行语音分类实验，支持：

- 纯音频分类
- 音频 + 文本融合分类

当前流程基于 `tasks/` 下任务目录，包含训练/验证、定期与最优检查点保存、训练后自动测试，以及 CSV 优先的分析输出。

## 功能概览

- 单一 CLI 入口运行实验
- 直接读取 `tasks/<task_name>/train.yml`、`val.yml`、`test.yml`
- 音频输入必选；可选文本分支用于融合
- 每 N 个 epoch 保存检查点，并始终保留最佳模型
- 训练后自动使用最佳检查点进行测试
- 先保存预测 CSV，再生成分析图表与统计

## 目录结构

```text
data_loader/         任务加载、批处理、YAML 解析
experiments/         运行时、训练、测试、分析
model/               音频编码器、融合模块、分类器
src/__main__.py      主 CLI 入口
tasks/               含 train/val/test YAML 的任务目录
```

## 依赖说明

项目默认需要可用的 Python 环境和以下核心依赖：

- `torch`
- `torchaudio`
- `librosa`
- `s3prl`
- `transformers`
- `PyYAML`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `shap`

若启用 TensorBoard，请额外安装 `tensorboard`；否则可使用 `--disable-tensorboard`。

## 任务数据格式

每个任务目录需包含：

- `train.yml`
- `val.yml`
- `test.yml`

YAML 示例：

```yaml
sample_0001:
  age: 56
  gender: female
  label: organic
  speaker_id: 0001
  text_keys:
    - /absolute/path/to/audio.wav
  texts:
    - age=56; gender=female
```

字段约定：

- `label`：监督标签
- `text_keys`：音频路径列表
- `texts`：文本负载列表
- 其它字段：可保留用于分析

## 运行入口

```bash
python src/__main__.py [OPTIONS]
```

列出可用任务：

```bash
python src/__main__.py --list-tasks
```

## 主要参数

- `--task-dir`：任务目录路径
- `--combine-mode`：`audio`、`concatenation`、`cross_attention`、`gated`、`film`
- `--feature-model`：音频预训练模型名（S3PRL 名称或 HuggingFace 模型 ID）
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--learning-rate`：学习率
- `--save-every`：每 N 个 epoch 存一次模型
- `--device`：`auto`、`cpu`、`cuda`
- `--text-fields`：从 `texts` 中选择文本字段
- `--text-equals`：按键值过滤文本负载
- `--demographic-embedding-dim`：年龄/性别/吸烟/饮酒每个字段的嵌入维度（总 demographic 维度为该值的 4 倍）
- `--enable-shap`：在测试阶段启用 SHAP 分析（默认关闭）
- `--disable-tensorboard`：关闭 TensorBoard

注意：

- 本实验始终需要音频输入。
- `--feature-model` 为必需（除非你在代码中另行改动）。
- 当 `--combine-mode=audio` 时，`--text-fields` 与 `--text-equals` 会被忽略。

## 示例命令

### 1) 纯音频

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode audio \
  --feature-model wav2vec_large \
  --epochs 50 \
  --save-every 10 \
  --batch-size 16
```

### 2) 音频 + 文本融合（concatenation）

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode concatenation \
  --feature-model wavlm_base \
  --text-fields age gender \
  --epochs 50 \
  --save-every 10 \
  --batch-size 16
```

### 3) 文本过滤（仅在融合模式下生效）

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode concatenation \
  --feature-model wavlm_base \
  --text-fields age gender \
  --text-equals gender=female
```

模型选择规则：

- S3PRL upstream 名称可直接使用。
- 也可以在 `--feature-model` 传入 HuggingFace 模型 ID（需安装 `transformers`）。

## 训练与检查点

每次运行会在 `.cache/runs/` 下创建带时间戳的目录。


- `results/predictions.csv`
- `results/test_summary.json`
- `results/training_summary.json`
- `results/history.csv`
- `results/analysis/confusion_matrix.csv`
- `results/analysis/confusion_matrix.png`
- `results/analysis/accuracy_by_label.csv`
- `results/analysis/accuracy_by_label.png`
- `results/analysis/shap_contribution_by_class.csv`
- `results/analysis/shap_abs_audio_demographic_by_class.png`
- `results/analysis/shap_abs_demographic_by_class.png`
- `results/analysis/shap_signed_audio_demographic_by_class.png`
- `results/analysis/shap_signed_demographic_by_class.png`

若安装了 `shap` 且测试样本包含可用元数据，会额外生成 SHAP 贡献度结果。SHAP CSV 会按类别给出音频特征、年龄、性别、吸烟和饮酒的贡献比例。
默认不会执行 SHAP。需要生成 SHAP 结果时请添加 `--enable-shap`。

## 文本输入机制

文本输入来自每条样本的 `texts`：

- `--text-fields all`：保留完整 payload
- 指定字段：仅提取匹配键值
- `--text-equals`：仅保留满足过滤条件的 payload

## 运行目录命名

```text
.cache/runs/<task_name>_<model_key>_<combine_mode>_<timestamp>
```

示例：

- `femh_wav2vec_large_audio_20260311_152902`
- `femh_wavlm_base_concatenation_20260311_142952`

## 快速自检

```bash
python src/__main__.py \
  --task-dir tasks/femh \
  --combine-mode audio \
  --feature-model wavlm_base \
  --epochs 1 \
  --batch-size 8 \
  --device cpu \
  --disable-tensorboard
```
