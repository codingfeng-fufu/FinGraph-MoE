# 当前实现简述

## 1. 整体链路

FinGraph-MoE 当前由两部分组成：**文档到图谱的前端链路**和**三个数值推理 expert**。

用户上传财经文档并输入问题后，系统先用 LLM 根据 query 抽取相关公司、年份、财务字段和关系，形成结构化图谱。随后 Router 根据 query 选择合适的 expert，再由 expert 完成数值计算。

```text
Document + Query
→ Query-conditioned Graph Extraction
→ Dense Router
→ SUM / COUNT / PREDICT Expert
→ Answer + Graph Trace
```

这套系统不是普通 RAG。普通 RAG 主要检索文本 chunk，再让 LLM 生成答案；这里是先把文档问题转成可计算图谱任务，再交给专门的 expert 做数值推理。

```text
文档 → 图谱 → expert 计算
```

## 2. SUM Expert

`SUM` 解决“总和是多少”的问题，例如总营收、总营业利润、总净利润。

输入是公司节点及其财务字段：

```text
company nodes: revenue / operating_profit / net_profit
query: target attribute
```

模型不是 GCN，而是 **query-conditioned gated pooling**。

它先根据 query 判断要加哪个字段，再给每个节点一个 soft gate，最后对目标字段做图级加和：

```text
node features + query
→ node gate
→ selected financial value
→ gated sum pooling
→ total value
```

这样设计的原因是：求和任务本质是加法，不一定需要 message passing。直接把“选择节点 + 加和字段”写进模型更稳定、更可解释。

## 3. COUNT Expert

`COUNT` 解决“满足条件的有多少个”的问题，例如收入大于某阈值的公司数量。

输入是公司节点和 query 中的条件：

```text
company nodes: revenue / net_profit / operating_profit
query conditions: attribute + operator + threshold
```

模型核心是 **soft comparator + calibration + add pooling**。

它把条件编码成结构化向量，对每个节点判断是否满足条件：

```text
node values + query condition
→ soft comparison
→ node score
→ add pooling
→ count
```

COUNT 的重点不是分类，而是保留候选全集，然后对每个节点做条件过滤。这样可以避免只抽取“看起来满足条件”的节点导致计数偏差。

## 4. PREDICT Expert

`PREDICT` 解决收入预测问题，例如预测某公司下一年的 revenue。

输入包括：

```text
target company history
peer company context
static financial features
baseline growth
```

当前上线的是效果最稳的 **tabular residual predictor**。

它不是直接预测收入，而是预测相对 baseline 的增长残差：

```text
history + peer context + static features
→ flat PredictV6 features
→ ElasticNet residual model
→ predicted growth
→ decoded revenue
```

同时保留了 GRU/LSTM 和 Kumo/RFM-style in-context 模型作为实验路线，但当前展示和部署用的是稳定性最好的 residual tabular expert。

## 5. Router

Router 负责决定 query 交给哪个 expert。

它使用：

```text
Sentence embedding
+ FAISS dense retrieval
+ cosine similarity
+ top-k weighted voting
```

不是简单关键词匹配，而是把用户 query 和训练式 query 编码成向量，检索相似问题，再根据相似问题所属 expert 投票选择。

## 6. 核心设计思想

不同 expert 对应不同数值任务结构：

```text
SUM     → 加和结构
COUNT   → 条件过滤结构
PREDICT → 时序预测结构
```

核心思想是：**把财经文档问题拆成结构化图谱任务，再由具有不同归纳偏置的 expert 完成数值推理。**
