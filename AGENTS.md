# AGENTS.md

本文件为进入 `PokieTicker` / `FinTrack` 仓库的 AI 编码代理提供**项目级默认指引**。目标是：先理解分层与实验语境，再做小而准、可验证的修改。

## 先阅读这些文档
- 总览与架构：[`README.md`](./README.md)
- 课程要求：[`项目七_基于AI算法的量化交易策略_实验要求.md`](./项目七_基于AI算法的量化交易策略_实验要求.md)
- 当前实验结果：[`量化交易策略结果.md`](./量化交易策略结果.md)
- 设计文档：[`docs/superpowers/specs/2026-04-25-ai-quant-strategy-backtest-design.md`](./docs/superpowers/specs/2026-04-25-ai-quant-strategy-backtest-design.md)

## 项目速览
- 后端：`FastAPI` + `SQLite` + `Polygon` + `LLM`
- 前端：`React 19` + `TypeScript` + `Vite` + `D3`
- 机器学习：特征工程、`XGBoost` / `LSTM`、推理、策略回测
- 仓库同时包含课程文档、实验结果与报告草稿；**除非用户明确要求，不要主动修改这些教学材料**

## 目录分层
- `backend/api/`：API 入口与路由；优先在现有 router 中扩展
- `backend/pipeline/`：新闻分层处理、对齐、相似度
- `backend/ml/`：特征、训练、推理、回测、模型元数据
- `backend/llm/`：LLM 客户端封装
- `backend/polygon/`：行情/新闻数据访问
- `frontend/src/`：页面、组件、样式、i18n
- `tests/`：测试，重点覆盖 ML/回测逻辑

## 工作原则
1. 先判断改动属于 `API / pipeline / ml / frontend / docs` 哪一层，再动手。
2. 优先最小改动，复用现有函数、路由、数据结构和模式。
3. 不打破分层：API 负责输入输出，pipeline 负责新闻处理，ml 负责特征/训练/推理/回测。
4. 不硬编码密钥、模型名、地址；继续走 `.env`、配置模块或现有 settings。
5. 涉及交易日对齐、T+N 收益、标签生成时，必须避免前视偏差。
6. 改接口或模型输出时，优先保持兼容；若必须变更，连同调用方和测试一起更新。

## 项目特有注意事项
- `backend/ml/models/` 下是现有模型产物与回测 JSON；**不要无故改动或覆盖**，除非用户明确要求重新训练。
- 修改 LLM 相关逻辑时，注意全局配置与 Layer1 独立配置可能不同。
- SQLite 适合当前项目，但新增 schema 或字段时优先检查 `backend/migration.py`。
- 后端 CORS 已允许 `5173` 和 `7777`，前端端口调整时注意同步验证。

## 常用验证命令
- 后端启动：`python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000`
- Python 测试：`pytest tests/ -v`
- 前端开发：`cd frontend && npm run dev`
- 前端构建：`cd frontend && npm run build`
- 前端检查：`cd frontend && npm run lint`

## 何时补测试
- 修改 `backend/ml/`、`strategy_backtest.py`、特征工程、收益计算、标签生成：尽量补或更新 `tests/`
- 修改 API 序列化字段：至少检查受影响路由与前端调用点
- 修改前端图表或数据字段：至少完成一次构建验证

## 仓库中的专用 agents
- 项目工程代理：[` .github/agents/pokieticker-project.agent.md`](./.github/agents/pokieticker-project.agent.md)
- 课程报告代理：[` .github/agents/report-writer.agent.md`](./.github/agents/report-writer.agent.md)

如果任务明显属于其中之一，优先使用对应专用 agent；本文件只保留所有任务都适用的共性约定。

## 默认输出要求
完成任务时，默认汇报：
1. 改了什么
2. 为什么这样改
3. 影响了哪些文件
4. 做了哪些验证
5. 还存在哪些风险或未验证项
