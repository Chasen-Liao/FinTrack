# 提升 ML Pipeline 的 AUC 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 降低当前标签噪声，并补齐以 `roc_auc` 为核心的实验与搜索流程，验证 MU/T+5 分类器能否在样本外排序能力上显著高于随机水平。

**Architecture:** 保持当前 `build_features -> search/train -> strategy_backtest` 主流程不变，在共享特征构建层新增更干净的未来收益标签，在实验与参数搜索层把 `roc_auc` 提升为一等指标，并用现有扩展窗口评估方式验证低噪声目标是否真的改善排序能力。在分类器样本外排名能力没有明显改善前，不修改策略层逻辑。

**Tech Stack:** Python、pandas、NumPy、scikit-learn metrics、XGBoost、pytest

---

## 文件结构

- 修改：`backend/ml/features.py`
  责任：定义可复用的未来收益列和目标标签列，供全部 XGBoost 训练路径共用。
- 修改：`backend/ml/experiment.py`
  责任：比较不同特征集、目标定义和模型时，输出包含 `roc_auc` 的扩展窗口结果。
- 修改：`backend/ml/model.py`
  责任：支持按 `roc_auc` 排序参数搜索结果，并补充统一多股票扩展窗口搜索能力。
- 修改：`backend/ml/train.py`
  责任：在现有 CLI 中暴露新的搜索指标和统一训练入口。
- 修改：`tests/ml/test_feature_enhancements.py`
  责任：验证未来收益标签生成和中性样本过滤逻辑。
- 修改：`tests/ml/test_model_search.py`
  责任：验证在请求时搜索排序会优先选择更高的 `roc_auc`，并覆盖新增统一搜索辅助逻辑。

### Task 1: 在共享特征构建层加入更低噪声的 T+5 标签

**Files:**
- Modify: `backend/ml/features.py`
- Test: `tests/ml/test_feature_enhancements.py`

- [ ] **Step 1: 先写失败测试**

向 `tests/ml/test_feature_enhancements.py` 添加下面的测试：

```python
from backend.ml.features import add_future_return_targets


def test_add_future_return_targets_creates_direction_and_big_move_labels():
    df = pd.DataFrame(
        {
            "close": [100.0, 103.5, 101.0, 106.0, 109.0, 112.0, 108.0],
        }
    )

    result = add_future_return_targets(df.copy())

    assert "future_return_t5" in result.columns
    assert "target_t5" in result.columns
    assert "target_big1_t5" in result.columns
    assert "target_up_big_t5" in result.columns
    assert "target_down_big_t5" in result.columns
    assert result.loc[0, "target_t5"] == 1
    assert result.loc[0, "target_big1_t5"] == 1
    assert result.loc[0, "target_up_big_t5"] == 1
```

- [ ] **Step 2: 运行测试，确认当前失败**

运行：

```bash
pytest tests/ml/test_feature_enhancements.py::test_add_future_return_targets_creates_direction_and_big_move_labels -v
```

预期：FAIL，并提示 `ImportError` 或 `AttributeError`，因为 `add_future_return_targets` 还不存在。

- [ ] **Step 3: 写最小实现**

在 `backend/ml/features.py` 中，于 `build_features` 之上新增下面的辅助函数，并把原来的内联目标构造替换成对它的调用：

```python
def add_future_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add future-return columns and lower-noise classification targets."""
    close = df["close"]

    df["future_return_t1"] = close.shift(-1) / close - 1
    df["future_return_t2"] = close.shift(-2) / close - 1
    df["future_return_t3"] = close.shift(-3) / close - 1
    df["future_return_t5"] = close.shift(-5) / close - 1

    df["target_t1"] = (df["future_return_t1"] > 0).astype(int)
    df["target_t2"] = (df["future_return_t2"] > 0).astype(int)
    df["target_t3"] = (df["future_return_t3"] > 0).astype(int)
    df["target_t5"] = (df["future_return_t5"] > 0).astype(int)

    df["target_big1_t5"] = (df["future_return_t5"].abs() > 0.03).astype(int)
    df["target_up_big_t5"] = (df["future_return_t5"] > 0.03).astype(int)
    df["target_down_big_t5"] = (df["future_return_t5"] < -0.03).astype(int)

    return df
```

然后把 `build_features` 里的这段代码：

```python
    # --- Targets: next-N-day direction ---
    df["future_return_t1"] = close.shift(-1) / close - 1
    df["future_return_t2"] = close.shift(-2) / close - 1
    df["future_return_t3"] = close.shift(-3) / close - 1
    df["future_return_t5"] = close.shift(-5) / close - 1
    df["target_t1"] = (close.shift(-1) > close).astype(int)
    df["target_t2"] = (close.shift(-2) > close).astype(int)
    df["target_t3"] = (close.shift(-3) > close).astype(int)
    df["target_t5"] = (close.shift(-5) > close).astype(int)
```

替换为：

```python
    # --- Targets: future-return labels shared by train/search/experiments ---
    df = add_future_return_targets(df)
```

- [ ] **Step 4: 再次运行测试，确认通过**

运行：

```bash
pytest tests/ml/test_feature_enhancements.py::test_add_future_return_targets_creates_direction_and_big_move_labels -v
```

预期：PASS

- [ ] **Step 5: 提交**

```bash
git add tests/ml/test_feature_enhancements.py backend/ml/features.py
git commit -m "feat: add lower-noise t5 target labels"
```

### Task 2: 把实验流程改成 AUC 优先，而不是准确率优先

**Files:**
- Modify: `backend/ml/experiment.py`
- Modify: `tests/ml/test_feature_enhancements.py`

- [ ] **Step 1: 先写测试**

向 `tests/ml/test_feature_enhancements.py` 追加下面的测试：

```python
from sklearn.metrics import roc_auc_score


def test_auc_metric_prefers_probabilities_over_hard_labels():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    assert roc_auc_score(y_true, y_prob) == 1.0
```

这个测试本身很简单，作用是把后续实验流程必须输出和依赖的指标锁定为概率排序指标。

- [ ] **Step 2: 运行现状验证**

运行：

```bash
pytest tests/ml/test_feature_enhancements.py::test_auc_metric_prefers_probabilities_over_hard_labels -v
python -m backend.ml.experiment MU
```

预期：
- pytest：PASS
- 实验命令输出的表格里仍然没有 `ROC-AUC` 列，也没有 `target_up_big_t5` 对应目标

- [ ] **Step 3: 写最小实现**

在 `backend/ml/experiment.py` 中先补导入：

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

然后把 `_expanding_cv` 替换为保留概率输出的实现：

```python
def _expanding_cv(X, y, n_folds=5, min_train=200, model_cls=None, model_kwargs=None):
    """Run expanding-window CV and return aggregate metrics."""
    n = len(X)
    if n < min_train + 20:
        return None

    test_size = (n - min_train) // n_folds
    if test_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        test_size = (n - min_train) // n_folds

    all_true, all_pred, all_prob = [], [], []

    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        if model_cls is None:
            model = XGBClassifier(
                max_depth=4,
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            model = model_cls(**(model_kwargs or {}))

        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        all_true.extend(y_te.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(y_prob.tolist())

    t = np.array(all_true)
    p = np.array(all_pred)
    prob = np.array(all_prob)
    acc = accuracy_score(t, p)
    base = max(t.mean(), 1 - t.mean())

    return {
        "n": len(t),
        "accuracy": round(acc, 4),
        "baseline": round(base, 4),
        "lift": round((acc - base) * 100, 1),
        "precision": round(precision_score(t, p, zero_division=0), 4),
        "recall": round(recall_score(t, p, zero_division=0), 4),
        "f1": round(f1_score(t, p, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(t, prob), 4),
    }
```

把目标映射扩展为：

```python
    targets = {
        "direction_t1": "target_t1",
        "direction_t2": "target_t2",
        "direction_t3": "target_t3",
        "direction_t5": "target_t5",
        "big_move_1pct": "target_big1_t1",
        "big_move_2pct": "target_big2_t1",
        "up_big_1pct": "target_up_big_t1",
        "up_big_3pct_t5": "target_up_big_t5",
    }
```

更新排序和输出表头：

```python
    results.sort(key=lambda x: (x["roc_auc"], x["lift"], x["f1"]), reverse=True)

    print(f"\n{'Target':<18} {'Features':<12} {'Model':<8} {'AUC':>6} {'Acc':>6} {'Base':>6} {'Lift':>6} {'F1':>6}")
    print("-" * 78)
    for r in results:
        lift_str = f"{r['lift']:+.1f}pp"
        print(
            f"{r['target']:<18} {r['features']:<12} {r['model']:<8} "
            f"{r['roc_auc']:5.3f} {r['accuracy']*100:5.1f}% {r['baseline']*100:5.1f}% "
            f"{lift_str:>6} {r['f1']*100:5.1f}%"
        )
```

- [ ] **Step 4: 运行实验，确认新指标已经出现**

运行：

```bash
pytest tests/ml/test_feature_enhancements.py -v
python -m backend.ml.experiment MU
```

预期：
- pytest：PASS
- 实验输出通过 `AUC` 列显式展示 `ROC-AUC`
- 输出中出现 `up_big_3pct_t5`

- [ ] **Step 5: 提交**

```bash
git add tests/ml/test_feature_enhancements.py backend/ml/experiment.py
git commit -m "feat: add auc-first experiment output"
```

### Task 3: 在训练 CLI 中把 `roc_auc` 变成真正可选的搜索目标

**Files:**
- Modify: `backend/ml/model.py`
- Modify: `backend/ml/train.py`
- Modify: `tests/ml/test_model_search.py`

- [ ] **Step 1: 先写失败测试**

向 `tests/ml/test_model_search.py` 追加下面的测试：

```python
def test_select_best_search_result_prefers_roc_auc_when_requested():
    results = [
        {
            "params": {"max_depth": 3},
            "param_count": 1,
            "accuracy_lift": 0.04,
            "f1": 0.58,
            "accuracy": 0.59,
            "roc_auc": 0.54,
        },
        {
            "params": {"max_depth": 4},
            "param_count": 1,
            "accuracy_lift": 0.02,
            "f1": 0.55,
            "accuracy": 0.57,
            "roc_auc": 0.61,
        },
    ]

    best = select_best_search_result(results, metric="roc_auc")

    assert best["params"] == {"max_depth": 4}
```

- [ ] **Step 2: 运行测试，确认失败**

运行：

```bash
pytest tests/ml/test_model_search.py::test_select_best_search_result_prefers_roc_auc_when_requested -v
```

预期：如果当前排序逻辑仍然默认偏向 `accuracy_lift`，或对 `roc_auc` 路径处理不完整，则这里应 FAIL。

- [ ] **Step 3: 写最小实现**

在 `backend/ml/model.py` 中保留 `select_best_search_result` 的通用性，但在 `backend/ml/train.py` 显式把它暴露成 CLI 能选的主指标。先增加：

```python
SEARCH_METRICS = ["accuracy_lift", "roc_auc", "f1"]
```

然后增加 CLI 参数：

```python
    parser.add_argument(
        "--metric",
        choices=SEARCH_METRICS,
        default="accuracy_lift",
        help="Primary metric used to rank expanding-window search candidates",
    )
```

把它传入 `search_xgboost_params`：

```python
                search = search_xgboost_params(
                    sym,
                    h,
                    n_folds=args.folds,
                    min_train=args.min_train,
                    metric=args.metric,
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
```

更新搜索输出，让 `roc_auc` 直接打印出来：

```python
                print(
                    f"  {sym}/{h} search ({args.metric}): "
                    f"lift={best['accuracy_lift']:.2%} "
                    f"auc={best['roc_auc']:.4f} "
                    f"acc={best['accuracy']:.1%} "
                    f"f1={best['f1']:.1%} "
                    f"params={json.dumps(search['best_params'], ensure_ascii=False)}"
                )
```

- [ ] **Step 4: 运行测试和一次真实搜索**

运行：

```bash
pytest tests/ml/test_model_search.py -v
python -m backend.ml.train --symbol MU --horizon t5 --search-params --metric roc_auc --market-benchmark --neutral-band 0.03
```

预期：
- pytest：PASS
- CLI 输出里出现 `search (roc_auc)`，并包含 `auc=...`

- [ ] **Step 5: 提交**

```bash
git add tests/ml/test_model_search.py backend/ml/model.py backend/ml/train.py
git commit -m "feat: add roc-auc ranking to xgboost search"
```

### Task 4: 增加统一多股票扩展窗口搜索，用于验证 AUC 是否来自更充分的样本

**Files:**
- Modify: `backend/ml/model.py`
- Modify: `backend/ml/train.py`
- Modify: `tests/ml/test_model_search.py`

- [ ] **Step 1: 先写测试**

向 `tests/ml/test_model_search.py` 追加下面的测试：

```python
from backend.ml.model import _compute_classification_metrics


def test_compute_classification_metrics_includes_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = _compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["roc_auc"] == 1.0
```

- [ ] **Step 2: 先运行这个基础测试**

运行：

```bash
pytest tests/ml/test_model_search.py::test_compute_classification_metrics_includes_roc_auc -v
```

预期：PASS。先把指标辅助函数锁住，再在统一搜索里复用它。

- [ ] **Step 3: 写最小实现**

在 `backend/ml/model.py` 中，在 `search_xgboost_params` 下方新增统一搜索辅助函数：

```python
def search_xgboost_params_unified(
    horizon: str = "t5",
    symbols: list[str] | None = None,
    n_folds: int = 5,
    min_train: int = 200,
    param_grid: dict[str, list] | None = None,
    metric: str = "roc_auc",
    include_market_benchmark: bool = False,
    neutral_band: float | None = None,
) -> dict:
    """Search XGBoost params on combined multi-ticker data with expanding-window CV."""
    target_col = f"target_{horizon}"
    df = build_features_multi(symbols, include_market_benchmark=include_market_benchmark)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough combined data for unified search ({len(df)} rows)"}

    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_neutral_samples(df, horizon, neutral_band)
    if df.empty or len(df) < min_train + 20:
        return {"error": f"Not enough combined rows after neutral filtering ({len(df)} rows)"}

    feature_cols = FEATURE_COLS_WITH_MARKET if include_market_benchmark else FEATURE_COLS
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    candidates = []
    for params in _make_param_grid(param_grid):
        cv_result = _expanding_window_cv(
            X=X,
            y=y,
            dates=dates,
            n_folds=n_folds,
            min_train=min_train,
            model_params=params,
        )
        if "error" in cv_result:
            continue
        candidates.append(
            {
                "params": params,
                "param_count": len(params),
                "accuracy": round(cv_result["accuracy"], 4),
                "baseline": round(cv_result["baseline"], 4),
                "accuracy_lift": round(cv_result["accuracy_lift"], 4),
                "precision": round(cv_result["precision"], 4),
                "recall": round(cv_result["recall"], 4),
                "f1": round(cv_result["f1"], 4),
                "roc_auc": round(cv_result["roc_auc"], 4) if cv_result["roc_auc"] is not None else None,
                "n_folds": cv_result["n_folds"],
                "total_predictions": cv_result["total_predictions"],
            }
        )

    if not candidates:
        return {"error": f"No valid unified search results for {horizon}"}

    best = select_best_search_result(candidates, metric=metric)
    return {
        "symbol": "UNIFIED",
        "horizon": horizon,
        "metric": metric,
        "include_market_benchmark": include_market_benchmark,
        "neutral_band": neutral_band,
        "candidate_count": len(candidates),
        "best_params": best["params"],
        "best_metrics": {
            "accuracy": best["accuracy"],
            "baseline": best["baseline"],
            "accuracy_lift": best["accuracy_lift"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
        },
    }
```

在 `backend/ml/train.py` 中导入该函数和统一训练函数：

```python
from backend.ml.model import train, train_unified, search_xgboost_params, search_xgboost_params_unified
```

增加统一搜索开关：

```python
    parser.add_argument("--unified", action="store_true", help="Use the combined multi-ticker training/search path")
```

然后在训练循环里增加统一搜索分支：

```python
            if args.search_params and args.unified:
                search = search_xgboost_params_unified(
                    horizon=h,
                    symbols=symbols,
                    n_folds=args.folds,
                    min_train=args.min_train,
                    metric=args.metric,
                    include_market_benchmark=args.market_benchmark,
                    neutral_band=args.neutral_band,
                )
                if "error" in search:
                    print(f"  UNIFIED/{h} search: {search['error']}")
                    continue
                best = search["best_metrics"]
                print(
                    f"  UNIFIED/{h} search ({args.metric}): "
                    f"lift={best['accuracy_lift']:.2%} "
                    f"auc={best['roc_auc']:.4f} "
                    f"acc={best['accuracy']:.1%} "
                    f"f1={best['f1']:.1%}"
                )
                continue
```

- [ ] **Step 4: 运行测试和一次统一搜索命令**

运行：

```bash
pytest tests/ml/test_model_search.py -v
python -m backend.ml.train --horizon t5 --search-params --unified --metric roc_auc --market-benchmark --neutral-band 0.03
```

预期：
- pytest：PASS
- CLI 输出 `UNIFIED/t5 search (roc_auc): ...`

- [ ] **Step 5: 提交**

```bash
git add tests/ml/test_model_search.py backend/ml/model.py backend/ml/train.py
git commit -m "feat: add unified auc search workflow"
```

## 验证清单

- 运行：

```bash
pytest tests/ml/test_feature_enhancements.py tests/ml/test_model_search.py -v
```

预期：PASS

- 运行：

```bash
python -m backend.ml.experiment MU
```

预期：输出可以直接比较 `up_big_3pct_t5` 和 `direction_t5` 的 `roc_auc`。

- 运行：

```bash
python -m backend.ml.train --symbol MU --horizon t5 --search-params --metric roc_auc --market-benchmark --neutral-band 0.03
python -m backend.ml.train --horizon t5 --search-params --unified --metric roc_auc --market-benchmark --neutral-band 0.03
```

预期：
- 单股票搜索给出更新后的 `best_metrics.roc_auc`
- 统一搜索给出跨股票学习是否改善排序能力的直接证据

## 成功标准

- 代码库拥有可复用的低噪声 T+5 标签，而不是只依赖 `future_return_t5 > 0`。
- 实验输出可以按 `roc_auc` 比较配置，而不是只看 accuracy lift。
- CLI 搜索可以显式优化 `roc_auc`。
- 仓库可以在同一套标签定义下，对比单股票与统一多股票扩展窗口 AUC。
- 任何“ AUC 已提升”的结论，都必须能被上面的验证命令直接复现。
