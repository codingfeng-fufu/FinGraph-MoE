<a id="top"></a>

# FinGraph-MoE

<p align="center">
  <a href="#english">English</a> |
  <a href="#chinese">中文</a>
</p>

---

<a id="english"></a>

## English

FinGraph-MoE is a prototype system for query-conditioned financial document reasoning.

The core idea is simple: a small financial document is first converted into a query-relevant graph, then routed to a specialized expert rather than answered by one generic model.

Current expert set:

- `SUM`: financial aggregation, such as total revenue or total net profit.
- `COUNT`: condition filtering and cardinality estimation.
- `PREDICT`: revenue forecasting from company history and peer context.

The current demo focuses on small documents. Large-document chunking and production-scale storage are intentionally out of scope for this stage.

### Architecture

```text
Uploaded document + user query
  -> LLM query-conditioned graph extraction
  -> FAISS dense router
  -> SUM / COUNT / PREDICT expert
  -> answer + graph/expert trace
```

Important design point: not every expert is a traditional message-passing GNN. The project uses graph-structured inputs, but each expert uses the inductive bias that best matches its task.

- `SUM` uses query-conditioned gated additive pooling.
- `COUNT` uses soft comparators, condition encoding, calibration, and add pooling.
- `PREDICT` currently deploys the strongest tabular residual artifact, with recurrent and Kumo/RFM-style variants kept as experiments.

### Repository Layout

```text
apps/demo/                 FastAPI demo app and Chinese workbench UI
src/moe_router/            document graph extraction, router, orchestration, benchmarks
src/realdata_experts/      active SUM / COUNT / PREDICT expert implementations
scripts/moe/               demo, router, and end-to-end benchmark entrypoints
scripts/realdata/          active real-data expert training scripts
tests/                     unit and smoke tests
docs/                      technical notes and presentation source files
data/demo/                 small demo assets safe to keep in git
```

Large local assets are intentionally ignored by git:

- `data/realdata_inputs/`
- `runs/`
- `archive/`
- `tmp/`
- legacy synthetic GNN prototypes and old experiment branches
- checkpoints, FAISS indexes, joblib artifacts, generated PPTs, screenshots, PDFs, and `.env`

See `docs/github_upload_checklist.md` before publishing.

### Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For LLM graph extraction, copy the environment template and fill in your own key:

```bash
cp .env.example .env
```

Example DashScope/Qwen config:

```dotenv
DASHSCOPE_API_KEY=your_key_here
LLM_GRAPH_MODEL=qwen-plus
LLM_GRAPH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Do not commit `.env`.

### Run The Demo

```bash
python scripts/moe/run_demo_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

If port `8000` is already occupied, run the same app on another port:

```bash
python -m uvicorn apps.demo.app:app --host 127.0.0.1 --port 8017
```

The demo can:

- upload a small financial document
- route a natural-language query to an expert
- extract a query-conditioned graph
- show the selected expert, retrieved examples, graph trace, and final answer

The app is deployable from a clean GitHub clone. Without local router/model artifacts, the web server still starts and the landing page is available. Expert inference will report that artifacts are not configured.

Check service readiness:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected clean-clone response:

```json
{
  "ready": false,
  "service_error": "..."
}
```

This is expected when the private/generated artifacts are not present.

### Full Expert Inference

Full SUM / COUNT / PREDICT inference requires generated local assets:

```text
data/moe_router_realdata_v1/router.joblib
data/moe_router_realdata_v1/router.faiss
data/moe_router_realdata_v1/training_queries.jsonl
runs/realdata_experts/sum_v2/best_model.pt
runs/realdata_experts/count_v2/best_model.pt
runs/realdata_experts/predict_tabular_elastic_net_flat_residual/artifact.joblib
```

These files are intentionally not committed because they are generated artifacts or depend on private/large data.

When local training data exists under `data/realdata_inputs/`, rebuild the router with:

```bash
python scripts/moe/build_moe_router_realdata.py
```

Train active experts with:

```bash
python scripts/realdata/train_sum_v2.py
python scripts/realdata/train_count_v2.py
python scripts/realdata/train_predict_tabular.py
```

After these assets exist, `/api/health` should return:

```json
{
  "ready": true,
  "service_error": null
}
```

### Current Benchmark Entry Point

```bash
python scripts/moe/run_current_realdata_benchmark.py
```

This evaluates the current real-data expert stack and writes benchmark outputs under `runs/`.

### Tests

```bash
pytest -q
```

The default test configuration is the GitHub-safe baseline. It avoids private data and generated model artifacts.

The current clean-release check covers:

- package installation with `pip install -e . --no-deps`
- default `pytest -q`
- Python syntax compilation
- FastAPI app import
- `/` and `/api/health` responses

### Technical Notes

- `docs/sum_expert_technical_note.md`: detailed SUM Expert input/output/network explanation.
- `docs/current_results_showcase.md`: current system summary and demo talking points.
- `docs/github_upload_checklist.md`: upload checklist and ignored local-only assets.

<p align="right"><a href="#top">Back to top ↑</a></p>

---

<a id="chinese"></a>

## 中文

FinGraph-MoE 是一个面向财经小型文档的 query-conditioned document-to-graph expert 系统。

系统不把所有问题都交给一个通用模型回答，而是先根据用户问题从文档中抽取最小必要图谱，再把问题路由给专门的数值推理 expert。

### 核心流程

```text
上传文档 + 用户问题
  -> LLM 根据 query 抽取图谱
  -> FAISS dense router 选择 expert
  -> SUM / COUNT / PREDICT expert
  -> 答案 + 图谱轨迹 + expert 解释
```

### Expert 设计

- `SUM`：求和类财经问题，例如总营收、总营业利润、总净利润。
- `COUNT`：条件过滤和公司数量统计。
- `PREDICT`：基于公司历史序列和同行上下文预测收入。

不是所有 expert 都是传统 message-passing GNN。项目使用图结构作为统一输入表示，但每个 expert 根据任务结构选择最合适的归纳偏置：

- `SUM`：query-conditioned gated additive pooling。
- `COUNT`：soft comparator + condition encoding + calibration + add pooling。
- `PREDICT`：当前部署版本是基于 PredictV6 flat features 的 tabular residual predictor，循环网络版本作为实验分支保留在本地。

当前阶段聚焦小型文档。大型文档切分、长期存储和生产级权限系统不在当前范围内。

### 目录结构

```text
apps/demo/                 FastAPI demo 和中文工作台前端
src/moe_router/            文档抽图、Router、编排和 benchmark
src/realdata_experts/      当前 SUM / COUNT / PREDICT expert 实现
scripts/moe/               demo、router、benchmark 入口
scripts/realdata/          当前真实数据 expert 训练脚本
tests/                     GitHub 安全的单元测试和 smoke tests
docs/                      技术说明和展示材料文字稿
data/demo/                 可提交的小型 demo 数据
```

以下内容不会上传到 GitHub：

- `.env`
- `data/realdata_inputs/`
- `data/moe_router_realdata_v1/`
- `runs/`
- `archive/`
- `tmp/`
- checkpoint、FAISS index、joblib artifact、生成的 PPT、截图、PDF、旧实验分支

### 1. 安装依赖

建议使用 Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 配置 LLM 抽图

复制环境变量模板：

```bash
cp .env.example .env
```

如果使用 DashScope / Qwen Plus，在 `.env` 中填写：

```dotenv
DASHSCOPE_API_KEY=your_key_here
LLM_GRAPH_MODEL=qwen-plus
LLM_GRAPH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

不要把 `.env` 提交到 GitHub。

### 3. 启动前端 Demo

```bash
python scripts/moe/run_demo_app.py
```

然后访问：

```text
http://127.0.0.1:8000
```

如果 `8000` 端口被占用，可以换端口：

```bash
python -m uvicorn apps.demo.app:app --host 127.0.0.1 --port 8017
```

然后访问：

```text
http://127.0.0.1:8017
```

### 4. 检查服务状态

```bash
curl http://127.0.0.1:8000/api/health
```

如果是刚从 GitHub clone 下来的干净版本，没有本地模型和 router artifact，返回一般是：

```json
{
  "ready": false,
  "service_error": "..."
}
```

这是正常情况。页面可以启动，但完整 expert 推理还不能跑。

### 5. 启用完整 Expert 推理

完整的 SUM / COUNT / PREDICT 推理需要本地存在以下生成产物：

```text
data/moe_router_realdata_v1/router.joblib
data/moe_router_realdata_v1/router.faiss
data/moe_router_realdata_v1/training_queries.jsonl
runs/realdata_experts/sum_v2/best_model.pt
runs/realdata_experts/count_v2/best_model.pt
runs/realdata_experts/predict_tabular_elastic_net_flat_residual/artifact.joblib
```

这些文件不会上传到 GitHub，因为它们是训练生成物，或者依赖本地真实数据。

如果本地有训练数据 `data/realdata_inputs/`，可以重建 router：

```bash
python scripts/moe/build_moe_router_realdata.py
```

训练 active experts：

```bash
python scripts/realdata/train_sum_v2.py
python scripts/realdata/train_count_v2.py
python scripts/realdata/train_predict_tabular.py
```

准备好这些 artifact 后，再访问：

```bash
curl http://127.0.0.1:8000/api/health
```

此时应该看到：

```json
{
  "ready": true,
  "service_error": null
}
```

### Benchmark 入口

```bash
python scripts/moe/run_current_realdata_benchmark.py
```

该脚本会评估当前真实数据 expert stack，并把 benchmark 输出写到 `runs/`。

### 测试

```bash
pytest -q
```

默认测试配置是 GitHub 安全版本，不依赖私有数据，也不依赖生成的模型 artifact。

当前 clean-release 检查包括：

- `pip install -e . --no-deps`
- `pytest -q`
- Python 语法编译
- FastAPI app import
- `/` 和 `/api/health` 响应检查

### 技术说明

- `docs/sum_expert_technical_note.md`：SUM Expert 的输入、输出和网络结构。
- `docs/current_results_showcase.md`：当前成果和 demo 讲解口径。
- `docs/github_upload_checklist.md`：GitHub 上传检查清单和本地忽略文件说明。

<p align="right"><a href="#top">返回顶部 ↑</a></p>
