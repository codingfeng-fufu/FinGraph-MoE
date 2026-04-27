# MoE-GNN 当前成果展示稿

## 1. 一句话定位

我们当前完成的是一个“查询驱动的文档到专家推理工作台”：

用户上传一份小型财务文档，系统根据问题抽取所需图谱，把问题路由给合适的专家模型，然后返回可解释的数值答案。

当前支持三类专家任务：

- `SUM`：按行业、年份、指标做聚合求和。
- `COUNT`：按数值条件统计满足条件的公司数量。
- `PREDICT`：基于目标公司历史和同行上下文预测收入。

## 2. 展示主线

建议展示时不要从模型开始讲，而是从产品链路开始：

1. 上传小型文档。
2. 输入自然语言问题。
3. LLM 根据 `document + query + routed expert` 抽取查询相关图谱。
4. Router 用 embedding + FAISS + cosine similarity 找相似训练问题，选择专家。
5. 专家模型执行数值推理。
6. 工作台展示证据、图谱、路由投票、相似问题和执行细节。

核心观点：

- 输入已经从“训练样本”推进到“真实文档”。
- 图谱不是固定全量抽取，而是 query-conditioned extraction。
- Router 已经升级成 dense embedding 检索，不再是纯 TF-IDF。
- Predict 专家已经接入当前真实数据上效果最好的 deployable 模型。

## 3. 当前系统架构

```text
小型文档
  -> 文档解析/上传会话
  -> Query-conditioned LLM Graph Extraction
  -> 结构化公司-年份财务图谱
  -> Query Rewrite
  -> Dense FAISS Router
  -> SUM / COUNT / PREDICT Expert
  -> 答案 + 证据 + 图谱 + 路由解释
```

### 文档到图谱

- 使用 `qwen-plus` 作为 LLM extraction 默认模型。
- 对每个 query 只抽取当前问题所需记录。
- 保留 evidence spans、source lines、单位、置信度和 assumptions。
- 对 `count` 保留候选全集，不只保留满足条件的行。
- 对 `predict` 保留目标公司历史和同行上下文。

### Router

- 当前 router backend：`dense_faiss`。
- embedding 模型：`intfloat/multilingual-e5-base`。
- embedding 维度：`768`。
- 检索记录数：`1648` 条训练式问题。
- 组合评分：dense similarity + lexical similarity + intent/anchor prior。
- 路由支持 query rewrite，包含 heuristic rewrite 和可选 LLM rewrite。

### 专家层

- `SUM v2`：图级聚合求和专家。
- `COUNT v2`：条件计数专家。
- `PREDICT tabular`：ElasticNet + flat PredictV6 features + residual growth。
- PREDICT 保留 v5 LSTM fallback，确保 artifact 不存在或异常时 demo 仍可运行。

## 4. 当前关键指标

### Router / MoE

- 路由器训练查询数：`1648`。
- 真实数据端到端评估中的路由准确率：`SUM 100%`、`COUNT 100%`、`PREDICT 100%`。
- Router 运行设备可使用 GPU embedding；FAISS index 可在 GPU 可用时迁移到 GPU。

注意：旧的 `moe_realdata_e2e.json` 中 PREDICT 指标来自早期 v2 predict 口径；当前工作台 PREDICT 已换成最新 tabular artifact。

### SUM 专家

- 数据规模：`train 400 / val 50 / test 150`。
- Test MAE：`0.00849`。
- Test RMSE：`0.01183`。
- Test R2：`1.0`。

解释方式：

SUM 任务基本已经稳定，适合作为工作台中“数值计算专家可被路由和执行”的确定性展示点。

### COUNT 专家

- 数据规模：`train 432 / val 75 / test 93`。
- Test rounded MAE：`0.3548` 家公司。
- Test exact match：`66.7%`。
- Test RMSE：`0.5648`。

解释方式：

COUNT 已经能完成条件计数推理。后续如果继续优化，重点不是重写路由，而是提升计数专家对复杂条件组合和边界样本的鲁棒性。

### PREDICT 专家

当前已集成到工作台的最佳真实数据模型：

- 模型：`ElasticNet`。
- 特征：`flat PredictV6 features`，包含历史序列、mask、同行上下文、静态特征。
- 目标：预测相对 peer/history baseline 的 residual growth。
- 数据规模：`train 700 / val 100 / test 200`。
- Test revenue MAPE：`6.50%`。
- Baseline revenue MAPE：`7.58%`。
- 相对 baseline MAPE 降低约：`14.2%`。
- Test growth MSE：`0.00638`。

集成状态：

- 工作台优先使用 `runs/realdata_experts/predict_tabular_elastic_net_flat_residual/artifact.joblib`。
- fallback：`predict_v5_lstm_h32_residual_015_e120/best_model.pt`。
- Smoke test 已确认策略为 `predict_tabular_elastic_net_flat_residual_scaled_growth`。

## 5. Kumo/RFM 风格探索结论

这一部分建议作为“预测任务下一阶段为什么这么做”的技术路线展示，不要说成已上线成果。

### v8：Kumo-style unseen-domain in-context generalization

实验设置：

- synthetic meta-generalization。
- train / val / test domain 完全不重叠。
- 目标是测试 unseen domain forecast 泛化能力。

结果：

- naive baseline MAPE：`5.07%`。
- query-only LSTM 平均 MAPE：`4.08%`。
- support DeepSets LSTM 平均 MAPE：`3.95%`。
- 相对 baseline 降低约：`22.1%`。

结论：

Kumo 类思路里最有价值的是“跨 domain episodic training + in-context support”，而不是简单堆一个更复杂的网络。

### v9：Relational-token RFM prototype

实验设置：

- 把 query entity、target history、support examples、schema/table/time/hop 编成 relational tokens。
- 用小型 Transformer 做 token-level relational reasoning。

结果：

- 最好 v9：`token_transformer_h64_l1`，test revenue MAPE：`4.32%`。
- 优于 naive baseline 和 support-mean baseline。
- 暂时弱于 v8 DeepSets。

结论：

Relational tokenization 是有方向的，但当前数据和预训练目标还不够支撑更复杂的架构。

### v10：Masked / pretrain 尝试

实验设置：

- 先在更多 synthetic domains 上做预训练，再 fine-tune forecasting。
- 目标是验证 foundation-style training 是否提升泛化。

代表结果：

- `pretrain48_h64_l1_seed31` test revenue MAPE：`4.29%`。
- baseline revenue MAPE：`5.05%`。

结论：

预训练方向有效，但目前还只是原型。下一步应该扩大 synthetic schema/task 覆盖，并设计更像 Kumo 的多任务预训练目标。

## 6. 工作台展示点

当前前端已经是中文产品化工作台，不只是调试页面。

建议现场展示这几个区域：

- 上传区：说明系统面向小型文档，而不是固定训练样本。
- 流程条：上传文档 -> 输入问题 -> 大模型抽取问题图谱 -> 路由到专家 -> 计算结果。
- 证据面板：展示 LLM extraction 的 source lines / evidence。
- 图谱视图：保留 query、filter、company、history、expert 等图结构。
- 执行结果：展示 prediction、answer、误差、专家策略。
- 路由投票：展示 dense retrieval 找到的相似训练问题。
- 调试 JSON：用于答辩时证明 API 返回了完整中间状态。

## 7. 现场演示脚本

启动：

```bash
python scripts/moe/run_demo_app.py
```

浏览器打开：

```text
http://127.0.0.1:8000/workbench
```

演示顺序：

1. 上传 `docs/demo_sample_financial_report.md`。
2. 先问 SUM：
   `请计算 2024 年 AI Infrastructure 行业公司的 revenue 总和`
3. 再问 COUNT：
   `请统计 2024 年 AI Infrastructure 行业 revenue >= 1400 的公司数量`
4. 最后问 PREDICT：
   `请预测 Nova Compute 在 2025 年的 revenue`
5. 对每个 query 指出：同一份文档生成了不同问题图谱。
6. 在 PREDICT 结果里强调：当前使用的是最新集成的 ElasticNet residual artifact。

## 8. 展示时可以强调的贡献

- 从 synthetic graph prototype 推进到了 real-data expert pipeline。
- 从固定图输入推进到了真实文档上传和 query-conditioned graph extraction。
- 从纯 lexical routing 升级到了 dense embedding + FAISS + cosine similarity。
- 建立了三个可路由专家：SUM、COUNT、PREDICT。
- PREDICT 做过 LSTM、GRU、DeepSets、attention、relational token、masked/pretrain、tabular residual 等多路线实验。
- 当前工作台已集成真实数据上最稳的 PREDICT artifact。
- 前端工作台保留图结构、证据链和路由解释，适合展示系统行为。

## 9. 风险和下一步

短期风险：

- 当前 LLM 抽图只针对小型文档，大型文档需要 chunking、分层抽取和证据合并。
- COUNT exact match 还有提升空间。
- 当前 PREDICT 上线的是真实数据上最稳的 tabular artifact，不是最终 GNN/RFM 方案。
- Kumo/RFM 类模型目前还是研究原型，尚未替换真实工作台 predict。

下一步建议：

- 做 query-conditioned graph extraction 的系统评测：抽取字段准确率、evidence 命中率、图谱完整率。
- 扩展文档处理到长文档：chunk retrieval -> LLM extraction -> graph merge。
- 保持 COUNT 当前路线，优先补复杂条件和边界样本测试。
- PREDICT 继续沿 Kumo-style 做 domain holdout + episodic pretraining，但上线模型先坚持“真实数据表现优先”。
- 把工作台里的 execution facts 做成更明确的“模型卡”，展示每个专家使用的 artifact、指标和 fallback。

## 10. 一分钟讲稿

这个系统的目标是把专家模型真正接到文档输入上。用户上传一份小型财务文档后，不是直接把全文丢给一个模型，而是先根据当前问题抽取一张查询相关的财务图谱。随后 router 使用 multilingual-e5 embedding、FAISS 和 cosine similarity 检索相似训练问题，并选择 SUM、COUNT 或 PREDICT 专家。专家完成计算后，工作台会同时展示答案、证据、图谱、路由投票和相似问题。

目前 SUM 和 COUNT 已经可以稳定跑完整链路。PREDICT 方面，我们实验了 LSTM、GRU、DeepSets、attention、relational token、masked/pretrain 和 tabular residual 等多条路线。当前上线到工作台的是真实数据上最稳的 ElasticNet residual 模型，test revenue MAPE 为 6.50%，比 peer/history baseline 的 7.58% 有明显提升。同时我们保留了 Kumo-style in-context learning 的实验路线，后续会把它作为提升泛化能力的方向继续推进。
