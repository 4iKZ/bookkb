# bookkb — 个人书库问答 CLI

[English](README.md) | 简体中文

![bookkb](docs/assets/hero-comic.webp)

把你的 PDF 电子书和 Markdown 笔记变成一个**本地、可引用**的知识库：
`ingest` 入库 → `ask` 提问，返回带**精确出处**（书名+页码 / 文件+章节）的原文证据。
CLI 只做检索，不做 LLM 合成——把证据喂给任何大模型（或 agent）再组织答案。

## 特性

- **本地离线**：embedding 用 ONNX 跑在本地，除模型下载外无外部 API 调用
- **中英双语**：`intfloat/multilingual-e5-small`，中英文混合检索
- **引用可追溯**：PDF 精确到页，Markdown 精确到文件+章节，每条结果带相似度分数
- **幂等入库**：按文件 sha256 去重，重跑只处理新增/变更
- **零服务**：一个包 + SQLite 单文件，无向量数据库、无 Web 服务

## 安装

```bash
pip install --user onnxruntime tokenizers numpy pypdf
pip install --user -e .          # 安装 bookkb 包 + console script
```

再准备 embedding 模型（`intfloat/multilingual-e5-small` 的 ONNX 版），目录结构：

```
<model-dir>/
  tokenizer.json
  onnx/model.onnx
```

两种方式提供模型路径（二选一）：

1. HuggingFace 缓存布局（默认）：`~/.cache/huggingface` 或 `HF_HOME` 下的
   `models--intfloat--multilingual-e5-small/snapshots/*/onnx/model.onnx`
2. 环境变量直接指定目录：`export BOOKKB_MODEL_DIR=/path/to/<model-dir>`

## 快速开始

```bash
# 1. 入库：PDF 文件/目录，或含 book/ + docs/ 的 markdown 仓库目录
bookkb ingest ~/books/纳瓦尔宝典.pdf
bookkb ingest ~/books/pdfs/
bookkb ingest ~/notes-repo/      # 需含 book/*.md（正文）与 docs/*.md（可选）

# 2. 提问：返回 top-k 证据（默认 8 条）
bookkb ask "What does Naval say about leverage?" -k 5
bookkb ask "失业了能领什么补助" -k 3
```

等效命令：`python3 -m bookkb ingest ...` / `python3 -m bookkb ask ...`。
旧版 `python3 bookkb.py ...` 已不再可用（单文件已移除，改用包结构）。

输出示例：

```
[1] 《纳瓦尔宝典-英文原版》第34页 (0.83)
"Fortunes require leverage. Business leverage comes from capital, people,
and products with no marginal cost of replication (code and media)."

[2] HowToLiveBetter/book/07-没钱的时候怎么活.md · 7. 没钱的时候怎么活 (0.91)
<chunk原文>
```

数据库位置：`data/bookkb.sqlite`（项目根目录下），可用 `BOOKKB_DB` 环境变量覆盖。

## 工作原理

![工作原理](docs/assets/how-it-works-comic.webp)

```
ingest:  PDF按页 / Markdown按二级标题 → 切 chunk → e5-small embedding
         → 存入 SQLite (documents, chunks)，embedding 为 float32 BLOB
ask:     问题同样编码 → NumPy 暴力算 cosine 相似度 → 返回 top-k 原文 + 出处
```

- e5 规范：入库文本加 `passage: ` 前缀，查询加 `query: ` 前缀，向量 L2 归一化
- chunk 上限 512 tokens（e5-small 上下文），PDF 单页超长按段落再分
- 万级 chunk 内暴力检索足够快，不引入向量数据库

## 引用格式

| 来源   | 引用格式 | 示例 |
|--------|----------|------|
| PDF    | 《书名》第N页 | 《纳瓦尔宝典-英文原版》第34页 |
| Markdown | 路径 · 章节标题 | HowToLiveBetter/book/05-不要浪费钱.md · 5. 不要浪费钱 |

## 已知局限

- 扫描版 PDF（无文本层）会被跳过并打印 warning，v1 不做 OCR
- 检索是纯向量相似度，无 rerank；同分或近似分时排序不完美
- 改动了已入库文件内容再重跑，替换语义未经充分测试；求稳可删库重建

## 项目文件

```
src/bookkb/     包源码
  __main__.py     python -m bookkb 入口
  cli.py          子命令装配（ingest / ask）
  ingest.py       文档发现 + 切块（PDF / markdown）
  embed.py        ONNX embedding（可注入测试 seam）
  store.py        SQLite 读写
  retrieve.py     检索 + 引用格式化
tests/          单测（切块、store 读写、检索排序）— 不依赖模型
docs/SPEC.md    v1 需求规格（中文）
AGENTS.md       给 agent 看的知识库使用说明（留在根目录）
data/           本地数据（gitignored）：SQLite 库
```

### 从旧版迁移

- 旧：`python3 bookkb.py ingest ...` → 新：`bookkb ingest ...`（或 `python3 -m bookkb ingest ...`）
- 旧：`python3 bookkb.py ask ...` → 新：`bookkb ask ...`（或 `python3 -m bookkb ask ...`）
- 数据库默认位置从脚本旁 `data/bookkb.sqlite` 改为当前工作目录下 `data/bookkb.sqlite`；
  仍可用 `BOOKKB_DB` 环境变量覆盖。

## License

MIT — 详见 [LICENSE](LICENSE)。
