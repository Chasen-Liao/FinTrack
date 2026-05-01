# AGENTS.md

本文件为进入 `PokieTicker` / `FinTrack` 仓库的 AI 编码代理提供项目级默认指引。目标只有两个：
- 先理解实验语境、目录分层与约束
- 再做小而准、可验证、尽量不外溢的修改

## 开始前先读
- 总览与架构：[`README.md`](./README.md)
- 课程要求：[`项目七_基于AI算法的量化交易策略_实验要求.md`](./项目七_基于AI算法的量化交易策略_实验要求.md)
- 当前实验结果：[`量化交易策略结果.md`](./量化交易策略结果.md)
- 设计文档：[`docs/superpowers/specs/2026-04-25-ai-quant-strategy-backtest-design.md`](./docs/superpowers/specs/2026-04-25-ai-quant-strategy-backtest-design.md)

## 仓库定位
- 后端：`FastAPI` + `SQLite` + `Polygon` + `LLM`
- 前端：`React 19` + `TypeScript` + `Vite` + `D3`
- 机器学习：特征工程、`XGBoost` / `LSTM`、推理、策略回测
- 仓库同时包含课程文档、实验结果与报告草稿；除非用户明确要求，不要主动修改这些教学材料

## 目录分层
- `backend/api/`：API 入口与路由；优先在现有 router 中扩展
- `backend/pipeline/`：新闻分层处理、对齐、相似度
- `backend/ml/`：特征、训练、推理、回测、模型元数据
- `backend/llm/`：LLM 客户端封装
- `backend/polygon/`：行情/新闻数据访问
- `frontend/src/`：页面、组件、样式、i18n
- `tests/`：测试，重点覆盖 ML 与回测逻辑

## 默认工作方式
1. 先判断任务属于 `api / pipeline / ml / frontend / docs` 哪一层，再决定改动位置。
2. 优先最小改动，复用现有函数、路由、数据结构和模式，不为了“更优雅”引入新架构。
3. 不打破分层边界：API 负责输入输出，pipeline 负责新闻处理，ml 负责特征、训练、推理与回测。
4. 不硬编码密钥、模型名、地址或环境差异；继续使用 `.env`、配置模块或现有 settings。
5. 变更接口、模型输出或数据结构时，优先保持兼容；如果必须变更，要同步更新调用方与测试。
6. 遇到歧义时先判断影响面：如果会影响架构、数据含义、用户体验、安全性或破坏现有产物，应先确认；否则可做最合理假设，但完成后要说明。

## 项目特有约束
- 涉及交易日对齐、`T+N` 收益、标签生成或回测时，必须显式避免前视偏差。
- `backend/ml/models/` 存放现有模型产物与回测 JSON；除非用户明确要求重新训练，不要覆盖、重写或清理。
- 修改 LLM 相关逻辑时，注意全局配置与 Layer1 独立配置可能不同，不要默认两者完全一致。
- 新增 SQLite schema、字段或迁移逻辑前，先检查 `backend/migration.py` 与现有初始化流程。
- 后端 CORS 当前已允许 `5173` 和 `7777`；前端端口、调用地址或部署方式有变动时，要一并验证。

## 验证要求
- 先把请求转换成可验证的成功标准，再实施修改。
- 修改 `backend/ml/`、`strategy_backtest.py`、特征工程、收益计算、标签生成时，尽量补或更新 `tests/`。
- 修改 API 序列化字段时，至少检查受影响路由和前端调用点是否一致。
- 修改前端图表、页面或数据字段时，至少完成一次构建验证；如条件允许，再补充实际页面检查。
- 完成后明确区分：哪些已经验证，哪些因环境、数据或时间限制尚未验证。

## 常用命令
- 后端启动：`python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000`
- Python 测试：`pytest tests/ -v`
- 前端开发：`cd frontend && npm run dev`
- 前端构建：`cd frontend && npm run build`
- 前端检查：`cd frontend && npm run lint`

## 专用 Agents
- 项目工程代理：[` .github/agents/pokieticker-project.agent.md`](./.github/agents/pokieticker-project.agent.md)
- 课程报告代理：[` .github/agents/report-writer.agent.md`](./.github/agents/report-writer.agent.md)

如果任务明显属于其中之一，优先使用对应专用 agent；本文件仅保留所有任务共用的默认约束。

## 默认交付说明
完成任务时，默认说明以下内容：
1. 改了什么
2. 为什么这样改
3. 影响了哪些文件
4. 做了哪些验证
5. 还存在哪些风险、假设或未验证项
