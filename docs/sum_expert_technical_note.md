# SUM Expert 技术说明

## 1. 任务定义

`SUM Expert` 用来处理求和类问题。

典型问题包括：

- 某份文档中所有公司的总营收是多少？
- 所有公司的营业利润之和是多少？
- 某个查询条件下，需要汇总的净利润是多少？

当前实现中，`SUM Expert` 支持三个目标字段：

```python
SUM_ATTRS = ("revenue", "operating_profit", "net_profit")
```

也就是：

- `revenue`：营收
- `operating_profit`：营业利润
- `net_profit`：净利润

核心思想是：文档先被抽取成结构化图谱，然后 `SUM Expert` 根据 query 判断应该加哪个字段，并对图中的节点进行图级聚合。

## 2. 原始输入格式

`SUM Expert` 的原始输入来自文档抽图结果。一个样本大致是下面的格式：

```python
{
    "graph": {
        "nodes": [
            {
                "id": "company_1",
                "type": "company",
                "attributes": {
                    "revenue": 1200.0,
                    "operating_profit": 160.0,
                    "net_profit": 95.0
                }
            },
            {
                "id": "company_2",
                "type": "company",
                "attributes": {
                    "revenue": 1800.0,
                    "operating_profit": 240.0,
                    "net_profit": 130.0
                }
            }
        ],
        "edges": []
    },
    "task": {
        "target": {
            "attribute": "revenue"
        }
    },
    "answer": 3000.0
}
```

这里最重要的部分有三个：

- `graph.nodes`：公司节点，每个节点有财务属性。
- `task.target.attribute`：当前 query 要求和的字段。
- `answer`：训练时使用的真实答案。

## 3. 模型输入格式

原始样本会被转换成 PyTorch Geometric 的 `Data` 对象。

```python
Data(
    x=...,
    node_values=...,
    edge_index=...,
    query_attr=...,
    y=...
)
```

具体张量如下：

```python
x: [num_nodes, 3]
node_values: [num_nodes, 3]
edge_index: [2, num_edges]
query_attr: [1, 3]
y: [1]
```

### 3.1 x

`x` 是缩放后的节点特征。

每个节点有三个特征：

```python
[
    signed_log1p(revenue),
    signed_log1p(operating_profit),
    signed_log1p(net_profit)
]
```

使用 `signed_log1p` 的原因是财务数据量级差异比较大，直接输入原始金额会导致训练不稳定。log 缩放可以压缩数值范围，同时保留正负号。

### 3.2 node_values

`node_values` 是原始财务数值。

每个节点对应：

```python
[
    revenue,
    operating_profit,
    net_profit
]
```

这里保留原始数值，是因为最终求和结果需要回到真实金额尺度，而不是 log 尺度。

### 3.3 query_attr

`query_attr` 是查询字段的 one-hot 表示。

```python
revenue          -> [1, 0, 0]
operating_profit -> [0, 1, 0]
net_profit       -> [0, 0, 1]
```

它告诉模型当前要加的是哪一列。

### 3.4 edge_index

`edge_index` 会从图谱中的边构造出来。

但是当前 `SUM Expert` 没有使用 `edge_index`。

也就是说，当前模型接收的是图结构输入，但中间网络没有做 GCN、GAT、GraphSAGE 那种邻居消息传递。

### 3.5 y

`y` 是训练标签，也就是真实求和结果。

```python
y = [answer]
```

## 4. 输出格式

模型输出两个结果：

```python
prediction, node_gates = model(...)
```

其中：

```python
prediction: [batch_size]
node_gates: [total_nodes]
```

### 4.1 prediction

`prediction` 是模型预测的求和结果。

它保持原始金额尺度，例如：

```python
prediction = 3000.0
```

### 4.2 node_gates

`node_gates` 是每个节点的软选择权重。

取值范围是 `0` 到 `1`。

可以理解为：

- 越接近 `1`，模型越认为这个节点应该被计入求和。
- 越接近 `0`，模型越认为这个节点不应该被计入求和。

这部分可以用于解释模型行为。

## 5. 中间网络结构

当前 `SUM Expert` 的模型是：

```python
SumV2Model(in_dim=3, hidden_dim=64)
```

它由四个主要模块组成：

- `node_encoder`
- `query_encoder`
- `gate_head`
- `correction_head`

整体结构如下：

```text
节点特征 x
  ↓
node_encoder
  ↓
node_hidden

查询字段 query_attr
  ↓
query_encoder
  ↓
query_hidden

node_hidden + query_hidden
  ↓
gate_head + sigmoid
  ↓
node_gates

node_values × query_attr
  ↓
selected_amount

node_gates × selected_amount
  ↓
global_add_pool
  ↓
base_sum

global_mean_pool(node_hidden) + query_attr
  ↓
correction_head
  ↓
correction

prediction = base_sum + 0.05 × correction
```

## 6. node_encoder

`node_encoder` 负责把每个公司节点的三个财务特征编码成隐藏向量。

结构是：

```python
node_encoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
)
```

输入：

```python
x: [total_nodes, 3]
```

输出：

```python
node_hidden: [total_nodes, 64]
```

作用是让模型理解每个节点自身的财务状态。

## 7. query_encoder

`query_encoder` 负责编码当前查询目标。

结构是：

```python
query_encoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
)
```

输入：

```python
query_attr: [batch_size, 3]
```

输出：

```python
query_hidden: [batch_size, 64]
```

作用是让模型知道当前是在问：

- 总营收
- 总营业利润
- 总净利润

## 8. gate_head

`gate_head` 是 `SUM Expert` 的关键模块。

它根据节点表示和查询表示，为每个节点生成一个 soft gate。

结构是：

```python
gate_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

输入是：

```python
concat(node_hidden, query_hidden_nodes): [total_nodes, 128]
```

输出是：

```python
node_gates: [total_nodes]
```

计算方式：

```python
gate_logits = gate_head(torch.cat((node_hidden, query_hidden_nodes), dim=-1))
node_gates = torch.sigmoid(gate_logits)
```

`node_gates` 表示每个节点是否应该参与当前 query 的求和。

初始化时，`gate_head` 最后一层 bias 被设置为 `4.0`。

这样初始状态下：

```python
sigmoid(4.0) ≈ 0.982
```

也就是说，模型一开始倾向于把大多数节点纳入求和。这对 SUM 任务是合理的，因为求和任务通常默认所有相关节点都应该被加进去，后续再由训练学习哪些节点需要被弱化。

## 9. selected_amount

模型用 `query_attr` 从 `node_values` 中选择当前要加的字段。

```python
selected_amount = (node_values * query_attr[batch]).sum(dim=-1)
```

如果当前 query 是求总营收：

```python
query_attr = [1, 0, 0]
```

那么：

```python
selected_amount = revenue
```

如果当前 query 是求总净利润：

```python
query_attr = [0, 0, 1]
```

那么：

```python
selected_amount = net_profit
```

## 10. global_add_pool

得到每个节点的 `node_gates` 和 `selected_amount` 后，模型做图级加和：

```python
base_sum = global_add_pool(node_gates * selected_amount, batch)
```

含义是：

```text
预测总和 = 所有节点的 soft gate × 目标字段值 之和
```

这一步是 `SUM Expert` 的核心归纳偏置。

它直接把“求和”这个数学结构写进了模型，而不是让模型从零学习任意复杂函数。

## 11. correction_head

除了主加和路径外，模型还有一个很小的校正项。

结构是：

```python
correction_head = nn.Sequential(
    nn.Linear(67, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

输入是：

```python
concat(global_mean_pool(node_hidden), query_attr)
```

输出是：

```python
correction: [batch_size]
```

最终预测：

```python
prediction = base_sum + 0.05 * correction
```

这里乘以 `0.05` 是为了限制 correction 的影响，避免校正项覆盖掉主要的加和结构。

换句话说，模型的主体仍然是可解释的加和路径，correction 只是补偿一些小偏差。

## 12. 它是不是 GNN

严格来说，当前 `SUM Expert` 不是传统 GNN。

原因是它没有使用：

- `GCNConv`
- `GATConv`
- `SAGEConv`
- message passing
- edge aggregation

它虽然使用了 PyTorch Geometric 的 `Data`、`Batch`、`global_add_pool` 和 `global_mean_pool`，但没有使用边上的邻居传播。

更准确的说法是：

```text
SUM Expert 是一个 graph-level gated pooling model。
```

或者：

```text
SUM Expert 是一个 query-conditioned DeepSets / gated additive pooling model。
```

也就是：输入是图，输出是图级数值，但中间网络不是 GCN。

## 13. 为什么 SUM 不一定要用 GCN

SUM 任务的本质是加法。

对于“总营收是多少”这类问题，最重要的是：

- 识别有哪些节点需要计入。
- 判断要加哪个字段。
- 把目标字段做稳定加和。

邻居消息传递不是这个任务的必要条件。

如果强行使用 GCN，可能会带来两个问题：

- 节点特征被邻居混合，反而破坏原始金额的可加性。
- 模型解释变差，因为最后的金额不再直接对应原始节点值。

所以当前设计更偏向结构化归纳偏置：

```text
用 gate 学习是否计入，用 pooling 保留求和结构。
```

这比直接套 GCN 更适合 SUM 任务。

## 14. 对外汇报口径

可以这样讲：

```text
SUM Expert 面向求和类财经问题。系统先将文档抽取为企业图谱，每个节点包含营收、营业利润、净利润等结构化字段。模型接收图谱和 query 字段，将节点特征与查询字段分别编码，再通过 query-conditioned gate 判断每个节点是否应该参与求和。最后，模型对被选择节点的目标字段执行可学习加和，并输出图级数值结果。
```

如果导师问“是不是 GCN”，可以回答：

```text
当前 SUM Expert 不是 GCN。它是图级 gated pooling 模型。我们保留了图结构输入和图级输出，但没有使用边上的消息传递。原因是 SUM 任务本身具有很强的加法结构，直接学习节点选择和字段加和比使用 GCN 更稳定、更可解释。
```

如果导师问“那图结构有什么意义”，可以回答：

```text
图结构的意义在于统一文档抽取后的表示。不同 expert 都接收图谱输入，SUM 只需要节点级字段和图级 pooling；COUNT 需要候选节点集合和条件过滤；PREDICT 可以利用时间、公司、行业等关系。也就是说，不是每个 expert 都必须使用同一种 GNN 网络，关键是根据任务结构选择合适的归纳偏置。
```

如果导师问“后面能不能换成 GNN”，可以回答：

```text
可以。我们可以把当前 SUM Expert 作为 gated pooling baseline，再补一个 GCN 或 GraphSAGE 版本作为对照实验。但从求和任务本身看，当前模型的可解释性和稳定性更强。
```

## 15. 一句话总结

`SUM Expert` 不是通用问答模型，也不是传统 GCN。它是一个面向财经求和任务设计的 query-conditioned gated pooling 模型：输入文档抽取出的企业图和查询字段，输出目标字段的图级总和，并能通过 node gate 解释哪些节点参与了求和。
