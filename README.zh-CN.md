<div align="center">

# 🌍 nanoGPT-Seis

[English](README.md) | [中文](README.zh-CN.md)

**从零训练一个面向地震科学的小型 GPT：从空文件夹到可推理模型，完整展示 LLM 预训练生命周期。**

采集 → 清洗 → 分词 → 建模 → 训练 → 推理，运行在 2× NVIDIA A30（48 GB）上。

</div>

![pipeline](assets/workflow.png)

<p align="center">六类免费数据源 → 爬取 → 清洗/去重 → 16k BPE →
113M GQA+RoPE decoder → 2-GPU DDP 训练 → 流式推理。</p>

---

nanoGPT-Seis 是一个教学型代码库。它的目标不是直接做出最强的地震大模型，而是把
预训练语言模型的每一个环节讲清楚：数据从哪里来，如何清洗和去重，tokenizer 如何
训练，Transformer 为什么这样设计，如何用两张 GPU 做 DDP 训练，以及如何做推理服务。
README 中的困惑度、显存、token 数等指标都来自实际实验。

语料混合了地震/地震学文本和通用英文文本。领域数据包括通过 Crossref+Unpaywall
获取的开放论文、arXiv/EarthArXiv 预印本、地震相关 Wikipedia 页面以及
“Earthquake Insights” Substack；通用数据包括 Wikipedia 和 FineWeb-Edu，用来改善
普通语言表达能力。总体约为 24% 领域文本 / 76% 通用文本。这样的规模让一个约 100M
参数的模型可以在单节点上完成完整训练闭环。

> **状态：**预训练生命周期已完成，包括数据采集、处理、tokenizer、训练和推理。

## 目录

1. [结果概览](#1-结果概览)
2. [快速开始](#2-快速开始)
3. [阶段 1：数据采集](#3-阶段-1数据采集)
4. [阶段 2：清洗与去重](#4-阶段-2清洗与去重)
5. [阶段 3：BPE tokenizer](#5-阶段-3bpe-tokenizer)
6. [阶段 4：模型结构](#6-阶段-4模型结构)
7. [阶段 5：训练](#7-阶段-5训练)
8. [阶段 6：推理](#8-阶段-6推理)
9. [代码结构](#9-代码结构)
10. [Scaling-law 实验](#10-scaling-law-实验)

---

## 1. 结果概览

| 项目 | 数值 |
|---|---|
| 语料 | 533,248 篇文档 · 485.7M words · **822.7M 训练 tokens**（约 2.4:1 通用:领域） |
| 模型 | **113M** 参数，decoder-only，GQA + RoPE + RMSNorm + SwiGLU |
| 硬件 | 2× NVIDIA A30（每张 24 GB），bf16，DDP；也可在单张 RTX 3090/4090 上运行，或通过减小 batch 在 12-16 GB 显存上运行 |
| 上下文长度 | **4096** tokens |
| 训练 | 8,000 iters，约 3.8 epochs，约 6.5 小时，约 2.9 s/iter |
| 通用文本 fluency | **0.997 bits/byte**；相比 domain-only base 的 1.527 降低约 35% |
| 推理 | KV-cache 流式生成，首 token 延迟约 176 ms，带 anti-repeat sampler |

<p align="center"><img src="assets/training_dynamics.png" width="80%"></p>

三个关键观察：

- **长上下文有帮助。** 在数据保持一致的对比实验中，4096 context 相比 1024 context
  将 domain-only 模型的 perplexity 从 10.93 降到 9.74，下降约 11%，而每步计算成本
  只增加约 26%。论文文本有长距离结构，1024 token 经常看不完整。
- **模型确实使用了长上下文。** 在 4096-token 窗口内，位置 2048-4096 的 token loss
  比位置 0-64 低约 25%，说明模型在利用前面几千个 token。
- **加入通用文本显著提升普通表达能力。** 加入 Wikipedia + FineWeb-Edu 后，通用文本
  bits/byte 相比纯论文模型降低约 35%，但领域 sharpness 有一定损失，这是 fluency 与
  specialization 的典型权衡。

### 数据混合：domain-only v1 → general + domain v2

纯论文模型能写出论文腔，但在普通英文上容易重复和失真。v2 加入约 540M tokens 的
Wikipedia + FineWeb-Edu，总训练 token 约 823M，训练约 3.8 epochs。用 bits-per-byte
评估时，v2 在通用文本上明显更流畅，同时保留了地震领域表达能力。

<p align="center"><img src="assets/corpus_composition.png" width="70%"></p>

<p align="center"><img src="assets/bpb_comparison.png" width="60%"></p>

---

## 2. 快速开始

### 2.1 直接试用预训练模型

预训练的 113M checkpoint 托管在 Hugging Face Hub：
[`jiazhe868/nanogpt_seis`](https://huggingface.co/jiazhe868/nanogpt_seis)。

```bash
# 环境：需要可用的 CUDA-12.4 PyTorch，见下方说明
conda activate nanogpt_seis
pip install -r requirements.txt

# 下载 checkpoint、tokenizer 和配置文件到 src/inference.py 默认寻找的位置
huggingface-cli download jiazhe868/nanogpt_seis \
    checkpoints/ckpt.pt \
    data/tokenized/tokenizer.json \
    data/tokenized/meta.json \
    configs/gpt120m_ctx4k.yaml \
    --local-dir .

python -m src.inference --prompt "The 2011 Tohoku earthquake"
```

### 2.2 从零复现预训练流程

```bash
# 环境
conda activate nanogpt_seis
pip install -r requirements.txt

# 地震领域数据源
python -m src.crawl.wikipedia        --max-pages 500
python -m src.crawl.fulltext         --per-journal 3000 --broad 30000 --workers 64
python -m src.crawl.preprints        --arxiv 3000 --eartharxiv 2000
python -m src.crawl.substack         --max 500

# 通用文本混合，用于提升普通语言流畅度
python -m src.crawl.general          --wiki-tokens 300000000 --fineweb-tokens 240000000

python -m src.process.build_corpus   --val-frac 0.005
python -m src.tokenizer.train_bpe    --vocab-size 16384
python -m src.tokenizer.encode

torchrun --standalone --nproc_per_node=2 \
    -m src.train --config configs/gpt120m_ctx4k.yaml

python -m src.inference --prompt "The 2011 Tohoku earthquake"
```

> **环境提醒。** 如果 PyTorch 的 CUDA 版本高于驱动支持，`torch.cuda.is_available()`
> 可能返回 `False` 并退回 CPU。请先运行：
>
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```
>
> 必须输出 `True`。本项目使用 `torch 2.6.0+cu124` 匹配 CUDA-12.5 驱动。

---

## 3. 阶段 1：数据采集

目标是从合法、免费、可复现的数据源构建地震科学语料。代码位于 `src/crawl/`。

| 数据源 | 模块 | 内容 | 方式 |
|---|---|---|---|
| 研究论文 | `fulltext.py` | 开放获取的地震论文全文 PDF | Crossref DOI → Unpaywall OA PDF → 下载 → 抽取文本 |
| 预印本 | `preprints.py` | arXiv + EarthArXiv 全文 | arXiv API + OSF/DOI → PDF |
| Wikipedia | `wikipedia.py` | 标题包含 earthquake 的页面 | MediaWiki API 文本抽取 |
| Substack | `substack.py` | Earthquake Insights 文章 | archive API + HTML 解析 |
| 通用文本 | `general.py` | Wikipedia + FineWeb-Edu | Hugging Face `datasets` streaming 到指定 token budget |

每个文档最终会被标准化为统一 schema：

```python
@dataclass
class Doc:
    source: str
    id: str
    title: str
    text: str
    url: str = ""
    date: str = ""
    extra: dict = field(default_factory=dict)
```

爬虫设计强调三点：

- **合法来源。** 论文全文通过 Unpaywall 找开放获取副本，优先 repository/green OA。
- **可恢复。** 已下载/已处理的 DOI 或文档 ID 会被跳过，长任务中断后可以继续。
- **礼貌并发。** 多线程下载时按 host 做节流，避免对单个服务器造成压力。

---

## 4. 阶段 2：清洗与去重

代码位于 `src/process/`。处理流程包括：

- Unicode 规范化、空白字符归一、控制字符清理。
- PDF 特有处理：断词修复，例如 `earth-\nquake → earthquake`。
- 去掉参考文献部分，减少 citation boilerplate。
- 长度过滤和质量过滤。
- MinHash/LSH 近重复去重，避免模型反复看到几乎相同的文本。
- 划分 train/val，并写入 `data/processed/`。

语料文件默认不会提交到 GitHub；用户可以通过爬虫重新生成。

---

## 5. 阶段 3：BPE tokenizer

代码位于 `src/tokenizer/`。项目使用 byte-level BPE：

- 词表大小：16,384。
- 特殊 token：`<|endoftext|>`。
- 编码结果以 `uint16` shards 保存，减少磁盘占用。

```bash
python -m src.tokenizer.train_bpe --vocab-size 16384
python -m src.tokenizer.encode
```

预训练模型对应的 `tokenizer.json` 和 `meta.json` 已托管在 Hugging Face Hub。

---

## 6. 阶段 4：模型结构

模型代码位于 `src/model/gqa_gpt.py`。这是一个 Llama-style decoder-only GPT：

| 组件 | 选择 | 作用 |
|---|---|---|
| Attention | GQA，12 query heads / 4 KV heads | 降低 KV cache 和推理显存 |
| 位置编码 | RoPE | 适合长上下文 |
| Norm | RMSNorm | 比 LayerNorm 更轻量 |
| MLP | SwiGLU | Llama 系列常用结构 |
| Token embedding | tied input/output embedding | 减少参数并改善语言建模 |
| Context | 4096 tokens | 覆盖论文长段落结构 |

主配置文件是 `configs/gpt120m_ctx4k.yaml`。

---

## 7. 阶段 5：训练

训练代码位于 `src/train.py`。默认训练设置：

- 2× NVIDIA A30，DDP。
- bf16 mixed precision。
- `torch.compile`。
- AdamW，cosine learning-rate schedule，warmup 400 iters。
- `batch_size=4`，`grad_accum=12`，2 GPUs，`block_size=4096`。
- 每步约 393,216 tokens。
- 8,000 iters，总计约 3.15B processed tokens。

训练命令：

```bash
torchrun --standalone --nproc_per_node=2 \
    -m src.train --config configs/gpt120m_ctx4k.yaml
```

显存不足时可以降低 `batch_size` 或提高 `grad_accum`，保持全局 token batch 大致不变。

---

## 8. 阶段 6：推理

推理代码位于 `src/inference.py` 和 `src/sample.py`。支持：

- 加载 checkpoint 和 tokenizer。
- KV-cache 流式生成。
- temperature / top-k / top-p sampling。
- repetition penalty 和 no-repeat ngram。
- 文本 perplexity / bits-per-byte 评估。
- context utilization 测试。

常用命令：

```bash
python -m src.inference --prompt "The 2011 Tohoku earthquake"
python -m src.inference --interactive
python -m src.inference --perplexity-text "A large subduction-zone earthquake..."
python -m src.inference --test
```

<p align="center"><img src="assets/generation_example.png" width="78%"></p>

---

## 9. 代码结构

```text
nanogpt_seis/
├── configs/                   # 模型/训练配置
├── src/
│   ├── crawl/                 # 数据采集
│   ├── process/               # 清洗、过滤、去重、划分
│   ├── tokenizer/             # BPE 训练和编码
│   ├── model/gqa_gpt.py       # Transformer 模型
│   ├── train.py               # DDP 训练循环
│   ├── inference.py sample.py # 推理和采样
│   ├── scaling/               # IsoFLOP scaling-law sweep
│   └── figures/               # README 图表生成脚本
├── assets/                    # README 图表
├── data/                      # 原始/处理后/tokenized 数据，默认 git-ignored
└── checkpoints/               # 权重和日志，权重默认 git-ignored
```

重新生成图表：

```bash
python -m src.figures.workflow
python -m src.figures.architecture
python -m src.figures.gqa_vs_mha
python -m src.figures.training_curves
python -m src.figures.corpus_composition
python -m src.figures.rope
CUDA_VISIBLE_DEVICES=0 python -m src.figures.generation_example
CUDA_VISIBLE_DEVICES=0 python -m src.figures.embedding_space
CUDA_VISIBLE_DEVICES=0 python -m src.figures.bpb_comparison
CUDA_VISIBLE_DEVICES=0 python -m src.figures.context_utilization
CUDA_VISIBLE_DEVICES=0 python -m src.figures.attention_map
```

---

## 10. Scaling-law 实验

代码位于 `src/scaling/`。这部分用小规模 IsoFLOP sweep 研究在固定计算预算下，模型大小
`N` 和训练 tokens `D` 如何权衡。

实验思路：

- 固定 context length 和全局 batch。
- 改变模型规模和训练 tokens。
- 每个预算下训练多个大小，找验证 loss 最低的点。
- 对最优点拟合 `N_opt ∝ C^a` 和 `D_opt ∝ C^b`。

运行方式：

```bash
python -m src.scaling.run_sweep --generate
python -m src.scaling.run_sweep --run --nproc 2
python -m src.scaling.fit
python -m src.figures.scaling_laws
```

<p align="center"><img src="assets/scaling_laws.png" width="92%"></p>

当前 sweep 得到的经验结果是：在这个小模型、小预算范围内，最优模型大小随计算预算增长
较快，`N_opt ∝ C^0.69`，`D_opt ∝ C^0.31`。这不应被解读为通用定律；它反映的是本项目
规模、语料、上下文长度和训练设置下的经验曲线。

---

## 致谢

本项目受到 Andrej Karpathy 的 nanoGPT 和 minimind 项目的启发。感谢 Crossref、
Unpaywall、arXiv、EarthArXiv/OSF、Wikipedia、FineWeb-Edu 和 Earthquake Insights
等开放科学基础设施。

## 数据与许可

- **代码**采用 MIT License，见 [LICENSE](LICENSE)。
- **原始语料不随仓库分发。** `data/` 默认被 git 忽略，需要通过爬虫重新生成。每个数据源
  保留自己的许可和使用条款。
- **模型权重**是派生 artifact，托管在 Hugging Face Hub，不提交到 GitHub。

## 引用

如果使用本项目，请引用：

```bibtex
@software{nanogpt_seis,
  title = {nanoGPT-Seis: the full LLM pretraining lifecycle on earthquake text},
  author = {jiazhe868},
  url = {https://github.com/jiazhe868/nanogpt-seis},
  license = {MIT}
}
```

## License

MIT — see [LICENSE](LICENSE).
