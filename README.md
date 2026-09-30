# bookkb — 个人书库问答 CLI

把你的 PDF 电子书和 Markdown 笔记变成一个**本地、可引用**的知识库：
`ingest` 入库 → `ask` 提问，返回带**精确出处**（书名+页码 / 文件+章节）的原文证据。
CLI 只做检索，不做 LLM 合成——把证据喂给任何大模型（或 agent）再组织答案。

## 特性

- **本地离线**：embedding 用 ONNX 跑在本地，除模型下载外无外部 API 调用
- **中英双语**：`intfloat/multilingual-e5-small`，中英文混合检索
- **引用可追溯**：PDF 精确到页，Markdown 精确到文件+章节，每条结果带相似度分数
- **幂等入库**：按文件 sha256 去重，重跑只处理新增/变更
- **零服务**：单个 `bookkb.py` + SQLite 单文件，无向量数据库、无 Web 服务

## 安装

```bash
pip install --user onnxruntime tokenizers numpy pypdf
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
python3 bookkb.py ingest ~/books/纳瓦尔宝典.pdf
python3 bookkb.py ingest ~/books/pdfs/
python3 bookkb.py ingest ~/notes-repo/      # 需含 book/*.md（正文）与 docs/*.md（可选）

# 2. 提问：返回 top-k 证据（默认 8 条）
python3 bookkb.py ask "What does Naval say about leverage?" -k 5
python3 bookkb.py ask "失业了能领什么补助" -k 3
```

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
bookkb.py    单文件 CLI（ingest / ask）
SPEC.md      v1 需求规格（中文）
AGENTS.md    给 agent 看的知识库使用说明
data/        本地数据（gitignored）：SQLite 库、入库的源文件、模型无关
```

## License

MIT — 详见 [LICENSE](LICENSE)。
