# ML Pipeline 改进设计：工程验证与报告自洽

> Status: Draft for review
> Date: 2026-05-03
> Scope: `backend/ml/`、`backend/ml/models/`、课程报告中的 ML 结果解释

## 1. 背景与问题判断

当前项目已经完成从新闻情绪、行情特征、XGBoost 训练到策略回测的完整闭环。课程报告中的代表性最佳策略为 MU / T+5 / threshold 0.5，回测年化收益率约 162.99%，最大回撤约 19.00%，满足课程要求中的收益和回撤目标。

但这个结果不能直接解释为“模型已经具备稳定预测能力”。现有证据显示，`direction_t5` 这类涨跌方向标签在扩展窗口验证下 ROC-AUC 接近 0.5，低噪声标签和 AUC 优先搜索只能把样本外排序能力小幅提升到约 0.50-0.54 区间。也就是说，策略收益、分类指标、买入持有对照需要分开解释：当前策略更像在强趋势样本上的择时和风险控制结果，而不是已经被证明的稳定 alpha。

因此，下一阶段 ML 改进的重点不是继续堆 XGBoost 参数，而是统一实验链路、降低预测目标噪声、修正时序验证风险，并让课程报告中的模型评价口径与工程证据一致。

## 2. 目标

本设计要达成五个目标：

1. 统一 `experiment.py`、`model.py`、`train.py` 和 `strategy_backtest.py` 的评估口径，避免实验结果和主训练结果来自不同特征、目标或验证方式。
2. 将主要预测目标从普通涨跌方向扩展到低噪声目标，例如 `target_up_big_t5`、neutral-band 过滤后的方向标签，或未来收益排序目标。
3. 修正 `features_v2.py` 中 TF-IDF/SVD 文本特征可能使用未来测试期词汇分布的问题，使文本特征在 walk-forward 验证中按训练窗口拟合。
4. 将 ROC-AUC、accuracy lift、基线对照、买入持有对照、跨股票分布和阈值稳定性作为共同验证标准。
5. 为课程报告提供更自洽的表述依据，明确区分“分类排序能力”“策略收益表现”和“强趋势行情下的风险控制”。

## 3. 非目标

本阶段不做以下事情：

1. 不覆盖、重写或清理 `backend/ml/models/` 中已有模型产物，除非后续实施计划明确要求重新训练。
2. 不承诺短期内一定提高策略收益。首要目标是提高实验可信度和解释一致性。
3. 不引入大规模深度学习重构。LSTM 或 Transformer 可以作为后续拓展实验，但不是本阶段主线。
4. 不把课程项目包装成生产级交易系统。报告应保留 close-to-close、手续费、样本期和数据规模等实验限制。

## 4. 推荐设计

### 4.1 目标定义层

保留现有 `target_t1`、`target_t5` 作为基线任务，但新增或提升低噪声目标的地位：

- `target_up_big_t5`：未来 5 个交易日上涨超过 3%。
- `target_big1_t5` / `target_big2_t5`：未来 5 个交易日绝对波动超过 1% 或 2%。
- `neutral_band` 过滤：训练方向分类时剔除未来收益绝对值过小的样本。
- 可选收益排序目标：保留未来收益本身，用于后续排名或分位数组合实验。

主训练入口不应只接受 `horizon=t1/t5` 推导出的 `target_{horizon}`。更合理的接口是显式接受 `target_col`，使实验、训练和回测能够使用同一目标定义。

### 4.2 特征层

现有新闻情绪、滚动新闻、事件类别、行业联动和技术指标可以继续作为主特征集。下一阶段重点不是盲目增加指标，而是清理特征评估方式：

- `features.py` 保持主流程稳定，继续提供可复用的基础特征和未来收益标签。
- `features_v2.py` 中的市场情绪、K 线形态和文本 SVD 特征应改成可插拔特征扩展。
- 文本特征不能在全样本上一次性 `fit_transform` 后再做时间序列验证。walk-forward 每一折必须只用训练窗口拟合 TF-IDF 和 SVD，再转换测试窗口。
- 市场基准特征应从“全股票平均 close”逐步替换或补充为更可解释的 `SPY`、`QQQ`、行业 ETF 或板块收益。

### 4.3 实验训练层

实验和主训练应共享同一套评估骨架：

```text
build_features
  -> choose feature set
  -> choose target_col / neutral_band
  -> walk-forward split
  -> fit model and optional feature transformer per fold
  -> collect probabilities
  -> compute AUC / lift / F1 / baseline
  -> optional strategy backtest from out-of-sample probabilities
```

推荐将 expanding-window / walk-forward 作为主验证方式。单次 80/20 holdout 可以保留为快速 sanity check，但不能作为报告中证明模型稳健性的主要证据。

参数搜索的排序指标应默认使用 `roc_auc`，并保留 `accuracy_lift`、`f1` 作为辅助排序条件。最终报告中应展示分布级结果，例如多股票平均 AUC、分位数、超过 0.5 的比例，以及代表性个股结果，而不是只展示 `strategy_best.json` 的赢家样本。

### 4.4 策略回测层

策略回测应继续保持 long/cash、close-to-close 和手续费假设，以保证课程项目可解释。但模型概率到交易信号的转换需要更严格：

- 阈值应在训练窗口或验证窗口中确定，不能只在全历史结果里挑最好看的阈值。
- 回测输出应同时报告策略收益、买入持有收益、最大回撤、交易次数、胜率和阈值稳定性。
- 如果模型 AUC 仍接近 0.5，报告应避免声称“模型预测能力强”，只能说明“该策略组合在样本期内满足课程指标”。
- 当买入持有收益高于策略收益时，应明确写成策略降低部分风险或控制回撤，而不是写成绝对收益更优。

### 4.5 报告解释层

课程报告应把结论拆成三层：

1. 工程实现层：系统完成数据获取、新闻情绪分析、特征工程、模型训练和策略回测闭环。
2. 策略表现层：MU / T+5 / threshold 0.5 在当前样本期内满足课程收益和回撤要求，但未必跑赢买入持有。
3. 模型能力层：扩展窗口 AUC 和 accuracy lift 显示当前分类排序能力仍偏弱，后续改进应聚焦低噪声目标、时序验证和跨股票稳健性。

这样的写法可以避免“收益高但 AUC 弱”的自相矛盾：收益表现是策略样本期结果，AUC 是模型样本外排序能力，两者相关但不是同一个证据。

## 5. 数据流

推荐的数据流如下：

```text
SQLite / stored JSON
  -> build_features(symbol)
  -> optional feature extension
  -> target selection
  -> walk-forward evaluator
  -> model search / model fit
  -> out-of-sample probability table
  -> threshold selection
  -> strategy backtest
  -> aggregate metrics
  -> report evidence table
```

关键约束是：任何需要从数据中学习参数的步骤，包括文本向量器、SVD、概率校准器、阈值选择器，都必须只在训练窗口或验证窗口内拟合，不能提前看见测试窗口。

## 6. 验证标准

本设计的成功标准不只看单个收益数字：

1. 对 MU / T+5，能复现实验中的旧目标 AUC 接近随机、低噪声目标小幅改善的结论。
2. 对多股票统一评估，输出 AUC 均值、中位数、分位数和超过 0.5 的股票比例。
3. 对每个候选目标，报告 accuracy、baseline、accuracy lift、precision、recall、F1 和 ROC-AUC。
4. 对策略回测，报告年化收益、累计收益、最大回撤、买入持有收益、交易次数和胜率。
5. 若改进后 AUC 仍弱，报告结论必须保持克制，不能把策略收益解释成稳定预测能力。
6. 测试覆盖至少包括目标标签生成、neutral-band 过滤、参数搜索排序、walk-forward 不泄漏测试窗口。

## 7. 风险与取舍

主要风险有四个：

1. 低噪声标签会减少样本量，可能提高 AUC 但降低训练稳定性。
2. 文本特征修正时序泄漏后，表面指标可能下降，但可信度更高。
3. 多股票统一训练可能抹平个股差异，需要保留单股票和跨股票两种视角。
4. 策略收益可能主要来自 MU 样本期强趋势，跨股票或换样本期后不一定保持。

这些风险不代表设计方向错误。相反，它们是报告需要主动披露的限制，也是后续实施计划中要验证的重点。

## 8. 后续实施边界

后续实施计划可以拆成四组任务：

1. 统一 `target_col` 接口，让实验、训练和回测使用同一目标定义。
2. 抽出 walk-forward evaluator，支持 fold 内特征转换、概率输出和统一指标计算。
3. 修正 `features_v2.py` 文本特征的时序拟合方式，并补测试。
4. 生成一份多股票聚合评估表，供课程报告引用。

实施时应保持最小改动：先让现有 XGBoost pipeline 的证据链变干净，再决定是否引入 LightGBM、LSTM 或更复杂的组合模型。
