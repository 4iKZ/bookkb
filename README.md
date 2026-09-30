![bookkb 吉祥物](docs/assets/hero.webp)

# 📚 bookkb — 你的私人书库小精灵

> 把 PDF 电子书和 Markdown 笔记喂给它，它就变成一只博闻强记的小精灵：
> 你随口一问，它翻开书、指着原文说——"喏，证据在这儿，第几页都给你标好了。"

`ingest` 投喂 → `ask` 提问，返回带**精确出处**（书名+页码 / 文件+章节）的原文证据。
CLI 只做检索、不替你动脑——把证据喂给任何大模型（或 agent）再组织答案。

## ✨ 小精灵的本事

- 🏠 **宅家办公**：embedding 用 ONNX 在本地跑，除了下载模型，不打一个电话给外网
- 🌏 **中英双语**：`intfloat/multilingual-e5-small`，中文英文混着问都行
- 🔖 **出处可查**：PDF 精确到页，Markdown 精确到文件+章节，每条结果还附相似度分数
- 🔁 **记性不乱**：按文件 sha256 去重，重跑一遍只处理新增/变更，老朋友不重复入库
- 🪶 **轻装上阵**：一个包 + 一个 SQLite 文件，没有向量数据库，没有 Web 服务，没有妖蛾子

## 🧰 把小精灵领回家

```bash
pip install --user onnxruntime tokenizers numpy pypdf
pip install --user -e .          # 安装 bookkb 包 + console script
```

再给它准备 embedding 模型（`intfloat/multilingual-e5-small` 的 ONNX 版），目录结构：

```
<model-dir>/
  tokenizer.json
  onnx/model.onnx
```

两种方式告诉它模型在哪（二选一）：

1. HuggingFace 缓存布局（默认）：`~/.cache/huggingface` 或 `HF_HOME` 下的
   `models--intfloat--multilingual-e5-small/snapshots/*/onnx/model.onnx`
2. 环境变量直接指路：`export BOOKKB_MODEL_DIR=/path/to/<model-dir>`

## 🚀 三步上手

```bash
# 1. 投喂：PDF 文件/目录，或含 book/ + docs/ 的 markdown 仓库目录
bookkb ingest ~/books/纳瓦尔宝典.pdf
bookkb ingest ~/books/pdfs/
bookkb ingest ~/notes-repo/      # 需含 book/*.md（正文）与 docs/*.md（可选）

# 2. 提问：返回 top-k 证据（默认 8 条）
bookkb ask "What does Naval say about leverage?" -k 5
bookkb ask "失业了能领什么补助" -k 3
```

等效命令：`python3 -m bookkb ingest ...` / `python3 -m bookkb ask ...`。

小精灵的回答长这样：

```
[1] 《纳瓦尔宝典-英文原版》第34页 (0.83)
"Fortunes require leverage. Business leverage comes from capital, people,
and products with no marginal cost of replication (code and media)."

[2] HowToLiveBetter/book/07-没钱的时候怎么活.md · 7. 没钱的时候怎么活 (0.91)
<chunk原文>
```

它的记事本放在 `data/bookkb.sqlite`（项目根目录下），可用 `BOOKKB_DB` 环境变量换地方。

## 🔍 小精灵的工作日常

![bookkb 工作流程](docs/assets/how-it-works.webp)

```
投喂(ingest):  PDF按页 / Markdown按二级标题 → 切 chunk → e5-small embedding
               → 存入 SQLite (documents, chunks)，embedding 为 float32 BLOB
提问(ask):     问题同样编码 → NumPy 暴力算 cosine 相似度 → 返回 top-k 原文 + 出处
```

- e5 门规矩：入库文本加 `passage: ` 前缀，查询加 `query: ` 前缀，向量 L2 归一化
- chunk 上限 512 tokens（e5-small 的上下文），PDF 单页超长按段落再分
- 万级 chunk 内暴力检索足够快，不请向量数据库这位贵客

## 🔖 引用格式

| 来源     | 引用格式       | 示例 |
|----------|---------------|------|
| PDF      | 《书名》第N页   | 《纳瓦尔宝典-英文原版》第34页 |
| Markdown | 路径 · 章节标题 | HowToLiveBetter/book/05-不要浪费钱.md · 5. 不要浪费钱 |

## 🙈 小精灵的短板（先说清楚）

- 扫描版 PDF（没有文本层）会被跳过并打印 warning，v1 不做 OCR——它看不懂图片里的字
- 检索是纯向量相似度，无 rerank；分数接近时排序可能不完美
- 改了已入库文件的内容再重跑，替换语义未经充分测试；求稳可删库重建

## 🗂️ 小精灵的房间布局

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

### 🧳 从旧版搬家

- 旧：`python3 bookkb.py ingest ...` → 新：`bookkb ingest ...`（或 `python3 -m bookkb ingest ...`）
- 旧：`python3 bookkb.py ask ...` → 新：`bookkb ask ...`（或 `python3 -m bookkb ask ...`）
- 数据库默认位置从脚本旁 `data/bookkb.sqlite` 改为当前工作目录下 `data/bookkb.sqlite`；
  仍可用 `BOOKKB_DB` 环境变量覆盖。

## 📄 License

MIT — 详见 [LICENSE](LICENSE)。
