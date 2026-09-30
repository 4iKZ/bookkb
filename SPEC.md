# bookkb — 个人书库问答 SPEC v1

## 背景
用户有 PDF 电子书 + 一个 markdown 知识库，想建成可检索的知识库。
以后用户直接在聊天里提问，由 orchestrator 调 CLI 取原文证据，再综合回答（CLI 不做 LLM 合成）。

## 数据源（已就位）
1. `~/workspace/your_files/纳瓦尔宝典-英文原版.pdf`（英文）
2. `~/workspace/your_files/巴比伦最富有的人-英文原版.pdf`（英文）
3. GitHub repo `https://github.com/eternity4719/HowToLiveBetter`（中文 markdown，主体在 `book/*.md`，另有 `docs/*.md` 长文）——实现时自行 `git clone` 到项目下（建议 `data/source/`）

## 功能需求

### 1. ingest（入库）
- 输入：一个或多个路径（PDF 文件 / 含 PDF 的目录 / 已 clone 的 repo 目录）。
- PDF：按页抽文本，记录页码。抽不出文本的页（扫描版）v1 跳过并打印 warning，不做 OCR。
- Markdown：`book/*.md` 每文件一个 doc，记录文件路径；长文件按二级标题切 chunk，记录章节标题。`docs/*.md` 同理。
- chunk 粒度：适配 embedding 上下文（e5-small 为 512 tokens）。PDF 以页为基本单位，单页超长按段落再分；每 chunk 元数据：书名/文件名 + 页码（PDF）或章节（md）。
- 去重：按文件 sha256 记录已入库，重复 ingest 跳过；repo 更新后重跑只处理新增/变更文件。
- 输出：打印入库统计（文件数、chunk 数、跳过数）。

### 2. ask（查询，返回证据）
- `bookkb ask "问题" [-k 8]`：返回 top-k 相关 chunks，纯文本格式：
```
[1] 《纳瓦尔宝典》第42页 (0.83)
<chunk原文>

[2] HowToLiveBetter/book/05-不要浪费钱.md · 章节标题 (0.79)
<chunk原文>
```
- 引用必须精确到页码或"文件+章节"，方便回答时标注来源。

### 3. 存储与检索
- sqlite 单文件：`data/bookkb.sqlite`（相对项目根目录，可用环境变量 `BOOKKB_DB` 覆盖）。
- 表：`documents(id, title, kind, path, sha256, ingested_at)`，`chunks(id, doc_id, page, section, text, embedding BLOB float32)`。
- 检索：cosine 相似度，numpy 暴力计算（万级 chunk 内足够，不引入向量数据库）。

### 4. Embedding（本地、离线）
- 模型：`intfloat/multilingual-e5-small`，用本机已有的 ONNX：
  `/home/hatch/.hf-models/models--intfloat--multilingual-e5-small/snapshots/*/onnx/model.onnx`
  （同目录有 `tokenizer.json`；`model_O4.onnx` 为量化版，可选用）。
- 推理：onnxruntime + tokenizers，中英皆可。
- e5 规范：doc 文本前缀 `passage: `，query 前缀 `query: `；向量 L2 归一化。
- 依赖安装：`pip install --user --break-system-packages onnxruntime tokenizers numpy pypdf`（按需，只装必要的）。

## 非目标（v1 不做）
- Web UI、用户系统、定时同步。
- OCR（扫描版 PDF）。
- CLI 内做 LLM 合成回答。

## 约束（ponytail 极简）
- 文件越少越好：能一个文件就别拆（单个 `bookkb.py` 带 ingest/ask 子命令优先）。
- stdlib 优先；第三方依赖只装必需的，README 列出。
- 每个子命令 `--help` 可用；注释只写 why 不写 what。

## 交付与自测（必须）
1. 在 `~/workspace/bookkb/` 实现（当前工作目录即项目根）。
2. 跑通：`ingest` 两本 PDF + clone 并 ingest HowToLiveBetter 的 `book/*.md` 和 `docs/*.md`。
3. 演示：`ask` 提 2 个问题——英文书里的（如 "What does Naval say about leverage?"），
   中文库里的（如"失业了能领什么补助"），把完整输出贴在最终返回里。
4. 最终返回：文件清单 + 依赖清单 + 自测输出 + 已知局限（一句话）。

## 完成标准
- ingest 幂等（重跑不重复入库）。
- ask 返回的引用精确、可追溯。
- 除 clone repo 外全程本地离线，无外部 API 调用。
