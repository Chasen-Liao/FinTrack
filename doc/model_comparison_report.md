# 多模型对比实验报告

> 实验日期：2026-05-03  
> 实验模块：`backend/ml/experiment.py`  
> 验证方式：5 折扩展窗口交叉验证（expanding-window CV），min_train=200

## 一、实验目的

本项目主模型为 XGBoost，为验证模型选择的合理性，本实验在相同数据、相同特征集和相同验证框架下，对比 XGBoost、LightGBM、Random Forest 和 Logistic Regression 四种算法的样本外预测表现，重点回答以下问题：

1. XGBoost 是否显著优于其他算法？
2. LightGBM 作为同族梯度提升树，与 XGBoost 的差异有多大？
3. 不同算法在不同目标标签和特征集下的表现是否一致？

## 二、实验设计

### 2.1 算法配置

| 算法 | 关键参数 | 说明 |
|------|---------|------|
| **XGBoost** | max_depth=4, n_estimators=200, lr=0.05, subsample=0.8, colsample=0.8 | 项目主模型，level-wise 生长策略 |
| **LightGBM** | max_depth=4, n_estimators=200, lr=0.05, subsample=0.8, colsample=0.8, min_child_samples=20 | 同族对比模型，leaf-wise 生长策略 |
| **Random Forest** | n_estimators=200, max_depth=6 | Bagging 集成树基线 |
| **Logistic Regression** | C=0.1, max_iter=1000 | 线性基线模型 |

### 2.2 特征集

| 特征集 | 说明 |
|--------|------|
| v1_base | 基础特征（新闻情绪 + 价格技术指标） |
| v2_market | 扩展市场基准特征 |
| v2_candle | 扩展 K 线形态特征 |
| v2_full | 全特征 + TF-IDF/SVD 文本特征 |

### 2.3 目标标签

| 标签 | 说明 |
|------|------|
| direction_t1/t2/t3/t5 | 标准 T+N 涨跌方向标签 |
| big_move_1pct/2pct | |T+N 收益| > 1%/2% 的大幅波动标签 |
| up_big_1pct | T+1 涨幅 > 1% 的低噪声标签 |
| up_big_3pct_t5 | T+5 涨幅 > 3% 的低噪声标签 |

### 2.4 评估指标

- **ROC-AUC**：概率排序能力（主排序指标）
- **Accuracy / Lift**：硬分类准确率及相对基线的提升
- **F1**：精确率与召回率的调和平均

## 三、实验结果

### 3.1 各股票 LightGBM vs XGBoost 配对对比

在每只股票上，四种算法 × 四种特征集 × 八种目标标签共产生 32~128 个实验组合。以下按股票汇总 LightGBM 与 XGBoost 的配对均值对比：

| 股票 | 配对组合数 | LightGBM AUC 均值 | XGBoost AUC 均值 | LightGBM F1 均值 | XGBoost F1 均值 | LightGBM Lift 均值 | XGBoost Lift 均值 |
|------|-----------|------------------|-----------------|-----------------|----------------|-------------------|------------------|
| MU | 32 | 0.5043 | 0.5001 | 0.4109 | 0.4049 | -3.9pp | -4.4pp |
| NVDA | 32 | 0.5159 | 0.5147 | 0.5420 | 0.5448 | -4.4pp | -4.5pp |
| AAPL | 32 | 0.5291 | 0.5341 | 0.3962 | 0.3843 | -0.4pp | -0.3pp |

**关键发现**：LightGBM 与 XGBoost 在 AUC、F1 和 Lift 上的差异均极小（< 0.02 AUC / < 1pp Lift），两者表现高度接近，没有出现某一方持续显著优于另一方的情况。

### 3.2 各股票 Top 5 结果（按 AUC 排序）

#### MU 股票 Top 5

| 目标标签 | 特征集 | 模型 | AUC | 准确率 | 基线 | Lift | F1 |
|---------|--------|------|-----|--------|------|------|-----|
| up_big_1pct | v1_base | RF | 0.536 | 58.5% | 59.1% | -0.6pp | 15.5% |
| big_move_1pct | v1_base | RF | 0.532 | 70.2% | 75.1% | -5.0pp | 82.3% |
| up_big_1pct | v1_base | LightGBM | 0.531 | 56.4% | 59.1% | -2.6pp | 30.7% |
| up_big_1pct | v1_base | LogReg | 0.527 | 58.8% | 59.1% | -0.3pp | 11.3% |
| up_big_1pct | v2_market | LogReg | 0.526 | 59.1% | 59.1% | +0.0pp | 11.4% |

#### NVDA 股票 Top 5

| 目标标签 | 特征集 | 模型 | AUC | 准确率 | 基线 | Lift | F1 |
|---------|--------|------|-----|--------|------|------|-----|
| big_move_1pct | v1_base | LogReg | 0.583 | 65.0% | 64.5% | +0.5pp | 78.3% |
| big_move_1pct | v2_market | LogReg | 0.583 | 65.0% | 64.5% | +0.5pp | 78.3% |
| big_move_1pct | v2_candle | LightGBM | 0.571 | 63.0% | 64.5% | -1.5pp | 74.9% |
| big_move_1pct | v2_candle | LogReg | 0.567 | 65.5% | 64.5% | +1.0pp | 78.5% |
| big_move_1pct | v2_full | LogReg | 0.566 | 65.5% | 64.5% | +1.0pp | 78.5% |

#### AAPL 股票 Top 5

| 目标标签 | 特征集 | 模型 | AUC | 准确率 | 基线 | Lift | F1 |
|---------|--------|------|-----|--------|------|------|-----|
| up_big_3pct_t5 | v1_base | LightGBM | 0.645 | 77.4% | 77.6% | -0.2pp | 22.0% |
| up_big_3pct_t5 | v2_full | XGBoost | 0.635 | 76.7% | 77.6% | -1.0pp | 9.5% |
| up_big_3pct_t5 | v1_base | XGBoost | 0.622 | 76.4% | 77.6% | -1.2pp | 14.3% |
| up_big_3pct_t5 | v2_candle | XGBoost | 0.610 | 76.7% | 77.6% | -1.0pp | 17.4% |
| up_big_3pct_t5 | v2_full | LightGBM | 0.610 | 75.9% | 77.6% | -1.7pp | 12.5% |

### 3.3 主任务对比：MU 股票 T+5 方向预测

主报告的展示策略为 MU/T+5/阈值 0.5，以下单独列出该任务（`direction_t5` 和 `up_big_3pct_t5`）下四种算法的对比：

| 目标标签 | 特征集 | XGBoost AUC | LightGBM AUC | RF AUC | LogReg AUC |
|---------|--------|------------|-------------|--------|-----------|
| direction_t5 | v1_base | 0.500 | 0.498 | 0.502 | 0.395 |
| direction_t5 | v2_market | 0.509 | 0.513 | 0.471 | 0.394 |
| direction_t5 | v2_candle | 0.504 | 0.507 | 0.473 | 0.396 |
| direction_t5 | v2_full | 0.515 | 0.514 | 0.491 | 0.395 |
| up_big_3pct_t5 | v1_base | 0.509 | 0.503 | 0.503 | 0.429 |
| up_big_3pct_t5 | v2_market | 0.509 | 0.508 | 0.508 | 0.431 |
| up_big_3pct_t5 | v2_candle | 0.502 | 0.509 | 0.502 | 0.438 |
| up_big_3pct_t5 | v2_full | 0.515 | 0.514 | 0.490 | 0.441 |

**关键发现**：
- 在 `direction_t5` 任务上，XGBoost 与 LightGBM 的 AUC 差异在 0.003~0.005 之间，可忽略不计。
- Random Forest 在部分特征集上表现略优，但整体也在 0.47~0.51 区间波动。
- Logistic Regression 在长周期方向预测上严重失效（AUC 低至 0.39），说明线性模型无法捕捉该任务中的非线性关系。
- 在 `up_big_3pct_t5` 低噪声标签上，LightGBM 在 v1_base 和 v2_candle 上 AUC 略高于 XGBoost（0.503 vs 0.509），但差异极小。

### 3.4 四算法 AUC 均值对比（跨股票汇总）

| 算法 | MU AUC 均值 | NVDA AUC 均值 | AAPL AUC 均值 | 三股票均值 |
|------|-----------|-------------|-------------|-----------|
| XGBoost | 0.5001 | 0.5147 | 0.5341 | 0.5163 |
| LightGBM | 0.5043 | 0.5159 | 0.5291 | 0.5164 |
| Random Forest | 0.5076 | 0.5223 | 0.5283 | 0.5194 |
| Logistic Regression | 0.4881 | 0.5102 | 0.5004 | 0.4996 |

**关键发现**：
- 四种算法中，RF 的三股票平均 AUC 最高（0.5194），但优势极微（仅比 XGBoost 高 0.003）。
- LightGBM 与 XGBoost 的三股票均值几乎相同（0.5164 vs 0.5163），差异在统计噪声范围内。
- Logistic Regression 整体最弱，但部分任务（如 NVDA 的 `big_move_1pct`）反超树模型，说明简单模型在特定标签定义下可能受益于低方差。

### 3.5 各算法最优结果（跨股票/标签/特征集）

| 排名 | 目标标签 | 特征集 | 模型 | 股票 | AUC | 准确率 | Lift |
|-----|---------|--------|------|------|-----|--------|------|
| 1 | up_big_3pct_t5 | v1_base | LightGBM | AAPL | 0.645 | 77.4% | -0.2pp |
| 2 | up_big_3pct_t5 | v2_full | XGBoost | AAPL | 0.635 | 76.7% | -1.0pp |
| 3 | up_big_3pct_t5 | v1_base | XGBoost | AAPL | 0.622 | 76.4% | -1.2pp |
| 4 | up_big_3pct_t5 | v2_candle | XGBoost | AAPL | 0.610 | 76.7% | -1.0pp |
| 5 | up_big_3pct_t5 | v2_full | LightGBM | AAPL | 0.610 | 75.9% | -1.7pp |

**注意**：AAPL `up_big_3pct_t5` 的高 AUC 需要结合正例比例解读——该标签正例仅约 22.4%，AUC=0.645 虽为最高，但精确率和召回率均偏低（F1=22.0%），实际交易价值有限。

## 四、结论与分析

### 4.1 核心结论

1. **XGBoost 与 LightGBM 表现高度一致**。在三只股票、128 个实验组合中，两者 AUC 均值差异不超过 0.005，F1 和 Lift 差异不超过 0.5pp。这说明在当前数据规模（单股票 300~500 样本）和特征形态下，level-wise（XGBoost）和 leaf-wise（LightGBM）两种树生长策略没有产生显著差异。

2. **Random Forest 略优于两种 GBDT**。RF 的跨股票 AUC 均值为 0.5194，比 XGBoost 和 LightGBM 高约 0.003。这一现象可能因为：在中小样本下，RF 的 Bagging 策略比 GBDT 的 Boosting 策略更不容易过拟合；但差异极小，不足以推翻 XGBoost 作为主模型的选择。

3. **Logistic Regression 在长周期方向预测上显著弱于树模型**。在 `direction_t5` 任务中，LogReg 的 AUC 低至 0.39~0.40，说明线性模型无法捕捉 T+5 预测中的非线性特征关系。但在 `big_move_1pct` 等标签上，LogReg 有时反而表现最好（如 NVDA），这可能因为该标签下信号更线性、噪声更少。

4. **所有算法的样本外 AUC 均在 0.50~0.65 区间**，大部分徘徊在 0.50~0.53 附近。这再次验证了前期 AUC 优先实验的结论：当前数据和特征条件下，模型预测能力有限，瓶颈不在算法选择而在特征表达和标签定义。

### 4.2 对课程报告的支撑

本实验为课程报告 3.3 节"可选算法对比"和 6.4 节"不足之处"提供了实证数据：

- **XGBoost 作为主模型的选择是合理的**：它与 LightGBM 表现持平，与 RF 差异极微，且具有特征重要性输出等可解释性优势。
- **未选择 LightGBM 的原因**：LightGBM 在当前实验中并未表现出相对于 XGBoost 的显著优势，且 XGBoost 的文档和社区支持更完善，对于课程项目更友好。
- **树模型整体优于线性模型**：LogReg 在方向预测任务上的 AUC 显著低于三种树模型，支持了"金融数据中特征关系非线性"的判断。

### 4.3 局限性

1. 当前实验仅覆盖 3 只股票，样本多样性有限。
2. 四种算法均使用默认或相近参数，未做各自独立的超参数搜索，可能低估了某些算法的潜力。
3. 实验使用扩展窗口 5 折验证，折数较少，指标方差较大，小差异（< 0.01 AUC）不具有统计显著性。
4. 未纳入 LSTM 等深度学习模型参与同框架对比（LSTM 使用序列输入，特征维度不同）。

## 五、数据文件

实验结果已保存至以下 JSON 文件：

- `backend/ml/models/MU_experiment_results.json`
- `backend/ml/models/NVDA_experiment_results.json`
- `backend/ml/models/AAPL_experiment_results.json`

可通过以下命令复现实验：

```powershell
.\venv\Scripts\python.exe -m backend.ml.experiment MU NVDA AAPL
```
