# slm-lab — Intent Router SLM 实验

用一个 **0.6B 小模型** 把有限范围内的用户输入准确分流，而不是做泛聊天。
第一版跑通完整链路：**小模型 → LoRA/SFT → 评测 → 再加 Teacher 蒸馏 → GGUF 量化 → 本地运行**。

```
Teacher: Qwen3-1.7B     →  生成/审核数据（蒸馏阶段）
Student: Qwen3-0.6B      →  LoRA/SFT  →  专用 0.6B SLM  →  GGUF 量化  →  llama.cpp
```

## 任务：工业设备 Intent Router

输入一句话，输出固定 JSON：

```json
{"intent":"diagnosis","domain":"equipment","need_rag":true,"need_graph":true,"missing_parameters":[]}
```

- `intent`: `knowledge_query` | `diagnosis` | `action` | `out_of_scope`
- `domain`: `equipment` | `maintenance` | `process` | `safety` | `unknown`
- `need_rag` / `need_graph`: 是否需要检索 / 图谱
- `missing_parameters`: 缺失参数，如 `["device_id"]`（接入 Agent 前置层的参数检查/重试/降级）

Schema 定义在 [scripts/schema.py](scripts/schema.py)，所有脚本共享。

## 目录

```
slm-lab/
├── data/            train/valid/test.jsonl（由脚本生成）
├── scripts/         generate_dataset / train_sft / evaluate / distill / merge_lora / export_gguf
├── configs/         sft.yaml / distill.yaml
├── models/          base / lora / final / distill
├── evaluation/      results.json / error_cases.jsonl
├── inference/       test_local.py（GGUF 本地测试）
└── requirements.txt
```

## 环境准备

```powershell
# CPU（最简，但训练很慢）
pip install -r requirements.txt

# GPU（推荐，训练快 50–150×）：装 CUDA 版 PyTorch，再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft datasets accelerate pyyaml
python -c "import torch; print(torch.cuda.is_available())"   # True = 能用 GPU
```

## 模型下载（Qwen3-0.6B）

本仓库**不含模型权重**，请自行下载到 `models/base/Qwen3-0.6B/`。

**ModelScope（国内直连，推荐）**
- 文件页：https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/files
- 命令行：
  ```python
  from modelscope import snapshot_download
  snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/base/Qwen3-0.6B')
  ```
- 或在文件页逐个下载，放进 `models/base/Qwen3-0.6B/`，至少包含：
  `config.json`、`generation_config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`

**Hugging Face（海外）**：仓库 `Qwen/Qwen3-0.6B`，可用 `huggingface_hub.snapshot_download` 或 `git lfs clone`。

> **Teacher（可选，蒸馏阶段才需要）**：GKD 蒸馏要求师生**同一词表**，用同系列的 `Qwen/Qwen3-1.7B`（别用 Qwen2.5），同样下载到 `models/base/Qwen3-1.7B/`。SFT 阶段不需要它。

> **受限/公司网络**：若代理强制认证会拦截 HF/ModelScope 的程序化下载（典型报错 `cpauth 401`），请用浏览器在上面文件页**手动下载**后放入目录，并在运行时打开离线开关：
> ```powershell
> $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
> ```

## 快速开始

```powershell
# 1) 生成数据（含正常/口语/模糊/多意图/缺参数/拼写错误/中英混合/越界）
python scripts/generate_dataset.py --train 2000 --valid 300 --test 500

# 2) 先评测 baseline（未训练），拿到对照基线
python scripts/evaluate.py --model models/base/Qwen3-0.6B --tag baseline

# 3) SFT + LoRA（CPU 用 configs/sft.yaml；GPU 用 configs/sft_gpu.yaml）
python scripts/train_sft.py --config configs/sft.yaml

# 4) 评测 SFT 后模型（LoRA 适配器）
python scripts/evaluate.py --model models/base/Qwen3-0.6B --adapter models/lora --tag sft

# 5) 合并 LoRA → 完整权重
python scripts/merge_lora.py --base models/base/Qwen3-0.6B --adapter models/lora --out models/final

# 6)（第二阶段）Teacher 蒸馏：TRL GKD（on-policy 分布级监督）
python scripts/distill.py --config configs/distill.yaml
python scripts/evaluate.py --model models/final --adapter models/distill --tag distill

# 7) 导出 GGUF 并量化，量化后必须重新评测
bash scripts/export_gguf.sh models/final models/final
python inference/test_local.py --gguf models/final/model-q4_k_m.gguf
```

## 评测矩阵

`scripts/evaluate.py` 输出到 [evaluation/results.json](evaluation/results.json)，错题落到 `error_cases.jsonl`。
不要只看 loss 下降——对比每一版：

| 指标 | Baseline | SFT | Distill | Q4/Q5/Q8 |
| --- | --- | --- | --- | --- |
| JSON Valid 率 | 100% | | | |
| Intent Accuracy | 0% | | | |
| Domain Accuracy | 0% | | | |
| 缺参数识别 | 99.4% | | | |
| 平均延迟 / P95 | 11675 / 16944 ms | | | |
| 模型大小 | ~1.4GB (BF16, 0.6B) | | | |

> Baseline = 未训练的 Qwen3-0.6B（`enable_thinking=False`，CPU）。它能产出**格式完全合法**的 JSON，但完全不懂意图分类——直接把用户整句塞进 `intent`、`need_rag/need_graph` 默认 false。这正是 SFT 要补的能力缺口。

工程判断不是「小模型比 GPT 强」，而是：**在明确任务上，92% 的准确率是否值得换取 10× 的延迟/部署优势。**

## 数据纪律

- 第一版规模：train 2000 / valid 300 / test 500，**test 完全独立**（生成器按 user text 去重后再切分）。
- 不只做正常案例，覆盖：模糊、多意图、缺参数、错参数、拼写错误、中英混合、口语、长文本、无关问题、越界请求。
- 想扩数据就往 [scripts/generate_dataset.py](scripts/generate_dataset.py) 加模板，或换成 Teacher 生成后写入 `data/raw/teacher_generated.jsonl`。

## CPU / GPU 与机器迁移

- 代码与硬件无关：`Trainer` 会自动使用可用的 CUDA 设备，评测脚本用 `device_map="auto"`。
- CPU 上 0.6B LoRA 训练很慢（约 90s/步，全程数小时）；GPU 上通常**分钟级**（快 50–150×）。
- 迁移到带 GPU 的机器：
  1. 拷贝整个 `slm-lab/`（**连 `models/base/Qwen3-0.6B/` 一起带走**，省去重新下载，尤其受限网络）。
  2. 装 CUDA 版 PyTorch（见「环境准备」），确认 `torch.cuda.is_available()` 为 `True`。
  3. 训练用 GPU 配置：`python scripts/train_sft.py --config configs/sft_gpu.yaml`
     （相对 CPU 配置：开 `bf16`、加大 batch、`max_length` 放回 512、可用全量 2000×3 epoch）。
  4. 其余命令（合并/评测/导出）完全一致。

## 后续升级方向

跑通后接回你现有系统：`FastAPI Gateway → Intent SLM 0.6B → {RAG(Qdrant) / Graph(Neo4j) / Action(Workflow)} → Evidence Pack → LLM`。
大模型负责复杂推理，SLM 负责高频确定性分流，RAG/Graph 负责事实，Workflow 负责执行。
