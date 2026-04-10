# Evolve Server

SkillClaw 的异步 **群体 Skill 进化引擎**。采用 **队列语义** 处理 session 数据：分布式 SkillClaw 客户端将交互 session 上传至共享存储的 `sessions/` 目录（充当队列），evolve_server 定期（或按需）**drain** 全部待处理 session，对每个 session 生成整体摘要，按 skill 维度聚合 session，最终由 LLM 在单次调用中自行决定是否进化（improve / optimize_description / skip）或创建新 skill。进化完成后 **ack**（删除已处理的 session 文件），进化产物更新至正式 skill 库。若进化过程失败，session 文件保留在存储中，下一轮自动重试。

## 队列缓存架构

共享存储的 `{group_id}/sessions/` 目录作为一个 **队列缓存**，分布式 SkillClaw 客户端是生产者，evolve_server 是消费者：

```
┌─────────────────┐     ┌─────────────────┐
│ SkillClaw 节点 1 │     │ SkillClaw 节点 2 │    ...（更多分布式节点）
│  (生产者)        │     │  (生产者)        │
└────────┬────────┘     └────────┬────────┘
         │  upload session JSON   │
         ▼                        ▼
┌──────────────────────────────────────────┐
│  Storage: {group_id}/sessions/ (队列)    │
│                                          │
│  sess_001.json  sess_002.json  ...       │
│                                          │
│  多个 SkillClaw 实例并发写入，            │
│  evolve_server 统一消费                   │
└────────────────────┬─────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Evolve Server       │
         │   (消费者)             │
         │                       │
         │  1. Drain: 读取全部    │
         │  2. Process: 进化流水线│
         │  3. Ack: 删除已处理    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Storage: skills/     │
         │  manifest.jsonl       │
         │  evolve_skill_registry│
         └───────────────────────┘
```

### 队列语义

| 操作 | 说明 |
|------|------|
| **Drain（排空）** | 每轮进化开始时，读取 `sessions/` 下所有 `*.json` 文件 |
| **Process（处理）** | 运行完整进化流水线（摘要 → 聚合 → 进化） |
| **Ack（确认）** | 进化成功后，从存储删除已消费的 session 文件 |
| **Nack（回退）** | 进化失败时，**不删除** session 文件，下一轮自动重试 |

这一设计保证：
- **无状态**：不依赖本地状态文件，evolve_server 可随时重启
- **幂等安全**：如果 session 在进化过程中失败，原始数据仍在存储中，下次重新处理
- **分布式友好**：多个 SkillClaw 节点并发上传 session，evolve_server 批量消费

## 目录结构

```
evolve_server/
├── __init__.py          # 包导出
├── __main__.py          # CLI 入口 (python -m evolve_server)
├── config.py            # EvolveServerConfig — 配置管理（多后端自动推断 + .env 加载）
├── constants.py         # DecisionAction、FailureType、SLUG 正则
├── llm_client.py        # AsyncLLMClient — OpenAI 兼容的异步 LLM 客户端（带重试和流式回退）
├── oss_helpers.py       # 存储读写工具函数（list/read/delete session、load/save manifest、fetch skill）
├── mock_bucket.py       # LocalBucket — 本地文件系统模拟 oss2.Bucket（同时用于 mock 和 local 后端）
├── skill_registry.py    # SkillIDRegistry — skill name→id 映射 + 版本追踪 + 历史记录
├── summarizer.py        # Session 双层预处理（程序化轨迹 + LLM 轨迹分析 + 元数据提取）
├── aggregation.py       # Session 级 Skill 维度聚合（按 session 整体对 skill 分组）
├── execution.py         # 当前唯一进化执行层：session 级 evolve/create + merge
├── utils.py             # 共享工具：JSON 解析、SKILL.md 渲染/解析、tool_call 压缩
├── server.py            # EvolveServer 核心编排 + 队列语义 + 冲突检测 + 编辑审计 + 调度
├── .env.example         # 配置模板
├── README.md            # 本文档
└── mock/                # Mock 测试数据（模拟存储目录结构）
    └── default/
        ├── manifest.jsonl
        ├── sessions/
        │   └── *.json
        └── skills/
            └── {skill-name}/SKILL.md
```

## 核心流程

`EvolveServer.run_once()` 执行一轮完整的进化周期：

```
共享存储 sessions/ 队列（分布式 SkillClaw 节点持续写入）
     │
     │  ① Drain: 读取全部待处理 session
     ▼
┌─────────────────────────────────────────┐
│  ② Session 双层预处理                   │
│  (summarizer.py)                        │
│                                         │
│  A. 程序化轨迹构建（零信息损失）:         │
│     逐步记录: user intent, skills used,  │
│     tool calls + outcomes, agent resp,   │
│     PRM score → session["_trajectory"]   │
│                                         │
│  B. LLM 轨迹感知分析（并行）:            │
│     - 因果链: read X → tried Y → fail Z │
│     - Skill 有效性: 哪些帮助/误导        │
│     - 关键转折点和失败根因               │
│     → session["_summary"]               │
│                                         │
│  C. 元数据提取:                          │
│    _skills_referenced: 所有引用的 skill   │
│    _avg_prm: 平均 PRM 分数              │
│    _has_tool_errors: 是否有工具错误       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  ③ Session 级 Skill 维度聚合            │
│  (aggregation.py)                       │
│                                         │
│  按 session 整体对 skill 分组:           │
│    session 中任意交互引用 skill X        │
│    → 该 session 归入 X 组               │
│    session 无任何 skill 引用             │
│    → 归入 __no_skill__ 桶               │
└─────────────┬───────────────────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌───────────┐   ┌───────────────────┐
│ Skill 组  │   │  No-skill 组      │
└─────┬─────┘   └────────┬──────────┘
      │                  │
      ▼                  ▼
┌─────────────┐  ┌─────────────────────┐
│ ④a 进化     │  │ ④b 进化             │
│ (单次 LLM)  │  │ (单次 LLM)          │
│             │  │                     │
│ LLM 看到:   │  │ LLM 看到:           │
│ - 当前 skill│  │ - session 轨迹      │
│ - session   │  │   (逐步操作路径)    │
│   轨迹(逐步 │  │ - session 分析      │
│   操作路径) │  │   (因果链+洞察)     │
│ - session   │  │ - 已有 skill 列表   │
│   分析(因果 │  │                     │
│   链+洞察)  │  │ 自行决定:           │
│ - 已有 skill│  │ create_skill → 新建 │
│   列表      │  │ skip → 不创建       │
│             │  │                     │
│ 自行决定:   │  │                     │
│ improve →   │  │                     │
│   改进内容  │  │                     │
│ optimize →  │  │                     │
│   改描述    │  │                     │
│ skip →      │  │                     │
│   不动      │  │                     │
└─────┬───────┘  └────────┬────────────┘
      │                   │
      └───────┬───────────┘
              │
              ▼
      ⑤ 上传进化产物到共享存储
         skills/{name}/SKILL.md
         manifest.jsonl 更新
         evolve_skill_registry.json 更新
              │
              ▼
      ⑥ Ack: 从存储删除已消费的 session 文件
```

## 各步骤详解

### Session 双层预处理 (`summarizer.py`)

对每个 session 做**两层信息提取**，确保后续进化 LLM 既有精确的操作轨迹又有高层分析：

#### A. 程序化轨迹构建 → `session["_trajectory"]`

`build_session_trajectory()` **不经过 LLM**，逐步（Step）遍历 session 中的每个交互，构建结构化文本：

```
[Step 1] PRM=0.9 | read_skills=['docker-compose-guide']
  User: Set up the database migration using alembic…
  Tools:
    bash(mkdir -p migrations) → ✓ cmd=mkdir -p migrations
    write_file(alembic.ini) → ✓
  Agent: Created migration directory and alembic config…

[Step 2] PRM=0.3 | read_skills=['docker-compose-guide']
  User: Run the migration now
  Tools:
    bash(alembic upgrade head) → ✗ [exit_code=1] table 'users' already exists
  Agent: The migration failed because the table already exists…

[Step 3] PRM=0.8
  User: Fix the migration error
  Tools:
    bash(alembic stamp head) → ✓
    bash(alembic revision --autogenerate -m "fix") → ✓
  Agent: Stamped current state and created a new revision…
```

每步保留的信息：
- **User intent**: 用户请求（前 200 字符）
- **Skills used**: 当步读取/注入的 skill
- **Tool calls + outcomes**: 工具名、参数摘要、成功/失败及返回内容（每步最多 5 条）
- **Agent response**: agent 回复摘要（前 200 字符）
- **PRM score**: 当步 PRM 得分

这一层是**零信息损失**的——操作序列、因果关系、错误上下文全部保留。

#### B. LLM 轨迹感知分析 → `session["_summary"]`

LLM 收到 session 全部交互（prompt 前 500 字符、response 前 400 字符、tool_calls 最多 6 条、tool_results 最多 6 条），生成 8-15 句的**轨迹感知分析**，重点是：

1. 因果链（read skill X → tried Y → failed Z → switched to W）
2. Skill 有效性评估（是否帮助、是否误导、缺少什么指导）
3. 关键转折点和失败根因
4. 工具使用模式
5. 最终结果质量

#### C. 元数据提取

- `_skills_referenced`: 从所有交互的 `read_skills` 和 `injected_skills` 中提取的 skill 名称集合
- `_avg_prm`: 所有非 None PRM 分数的平均值
- `_has_tool_errors`: 是否任何交互存在工具错误

### Session 聚合 (`aggregation.py`)

聚合单位是 **session**。遍历所有 session 的 `_skills_referenced`：

- 如果 session 引用了 skill X → 该 session 归入 X 组
- 一个 session 引用了 skill A 和 B → 同时出现在 A 组和 B 组
- 一个 session 无任何 skill 引用 → 归入 `__no_skill__` 桶

### Skill 组进化 (`execution.py` → `evolve_skill_from_sessions`)

对每个 skill 组，**一次 LLM 调用**同时完成决策和执行。LLM 收到双层证据：
- **轨迹**（`_trajectory`）: 每个 session 的精确操作路径——逐步工具调用、参数、成功/失败、PRM 分数
- **分析**（`_summary`）: LLM 对每个 session 的因果链分析和 skill 有效性评估
- 当前 skill 的完整内容（name / description / content）
- 已有 skill 名称列表

LLM 自行决定并产出结果：

| 决策 | 含义 | 产出 |
|------|------|------|
| `improve_skill` | skill 内容需要改进 | 返回编辑后的 skill（name/description/content/edit_summary） |
| `optimize_description` | skill 正文不动，描述需要更精准 | 返回新的 description |
| `skip` | skill 运行良好或证据不足 | 仅返回 rationale |

编辑原则（improve 路径）：
- 默认做局部编辑，不重写
- 保留原始结构、标题顺序、术语和有效指导
- 仅当证据明确支持时才修改
- 不改变 API 契约、端口、输出路径

**编辑审计**: 上传前检查 improve 结果的 Markdown section 变更，如果检测到 rewrite-like 行为（删除 section 或超过 50% section 被修改），拒绝该次进化。

### No-skill 组处理 (`execution.py` → `create_skill_from_sessions`)

同样**一次 LLM 调用**完成：

| 决策 | 含义 | 产出 |
|------|------|------|
| `create_skill` | 发现可复用的模式 | 返回新 skill（name/description/content） |
| `skip` | 无可操作的模式 | 仅返回 rationale |

## 冲突检测与 Merge

每个 skill 上传前经过冲突检测流程 (`_resolve_and_upload`):

1. **SHA 对比**: 计算进化后 SKILL.md 的 SHA-256，与 registry 中记录的 `content_sha` 对比
2. **无冲突**: 直接上传
3. **有冲突**: 下载存储中的现有版本，调用 LLM merge (`execute_merge`) 合并两个版本
4. **Merge 失败**: 保留新版本（覆盖写入）

## Skill ID 与版本管理 (`skill_registry.py`)

- **ID 生成**: `SHA-256(skill_name)[:12]`，确定性哈希，同名永远同 ID
- **版本追踪**: 每次内容变更自增版本号，记录 `content_sha` + `action` + `timestamp`
- **历史上限**: 每个 skill 最多保留 20 条历史记录
- **持久化**: `{group_id}/evolve_skill_registry.json` 存储在共享存储上

## 存储后端 (`config.py` + `mock_bucket.py`)

支持三种存储后端，通过 `EVOLVE_STORAGE_BACKEND` 或自动推断确定：

| 后端 | 说明 | 关键配置 |
|------|------|---------|
| `local` | 本地文件系统 | `EVOLVE_STORAGE_LOCAL_ROOT` 或 `--local-root` |
| `oss` | 阿里云 OSS | `EVOLVE_OSS_ENDPOINT` + `EVOLVE_OSS_BUCKET` + 密钥 |
| `s3` | S3 兼容存储 | `EVOLVE_STORAGE_ENDPOINT` + `EVOLVE_STORAGE_BUCKET` + 密钥 |

## LLM 客户端 (`llm_client.py`)

`AsyncLLMClient` 封装 OpenAI SDK，通过 `asyncio.to_thread` 实现异步调用：

- **重试**: 最多 6 次，指数退避 + 随机抖动
- **特殊模型适配**: 对 `kimi-k2.5` 等模型强制 `temperature=1`
- **流式回退**: 如果 API 返回 400 "Stream must be set to true"，自动切换为 SSE 流式请求
- **温度移除回退**: 如果 API 不支持 temperature 参数，自动移除后重试

## 错误恢复

队列语义的核心安全性保证：**session 只在进化全部成功后才被删除**。

- **正常流程**：drain → summarize → aggregate → evolve → ack (删除) → 记录 history
- **异常流程**：drain → process 失败 → session 保留 → 下一轮 drain 重新拿到 → 自动重试
- **服务重启**：由于 session 在共享存储上，重启后 drain 照常获取未处理的 session
- **编辑审计保护**: improve 动作如果被检测为 rewrite-like，会被拒绝

## 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv

# 如果需要连接阿里云 OSS:
pip install oss2

# 如果需要 S3 兼容存储:
pip install boto3

# 如果需要 HTTP trigger 模式:
pip install fastapi uvicorn
```

### 2. 配置

```bash
cd evolve_server
cp .env.example .env
# 编辑 .env，填入存储和 LLM 的密钥
```

### 3. 启动

```bash
# 定时模式（默认每 10 分钟执行一轮）
python -m evolve_server --use-skillclaw-config

# 单次运行
python -m evolve_server --once --use-skillclaw-config

# 带 HTTP 触发端点
python -m evolve_server --port 8787 --use-skillclaw-config

# 使用本地存储后端
python -m evolve_server --once --local-root /path/to/data

# Mock 测试模式
python -m evolve_server --mock -v
```

## HTTP API

启用 `--port` 后可用以下端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/trigger` | 立即执行一轮进化，返回 summary JSON |
| `GET` | `/status` | 查看队列中待处理 session 数、已注册 skill 及版本 |
| `GET` | `/health` | 健康检查 |

## CLI 参数一览

| 参数 | 说明 |
|------|------|
| `--once` | 单次运行后退出 |
| `--mock` | 使用本地 mock/ 目录替代远程存储（自动单次运行） |
| `--mock-root PATH` | 指定自定义 mock 目录 |
| `--port PORT` | 启用 HTTP server |
| `--interval SEC` | 定时间隔（秒，默认 600） |
| `--model MODEL` | LLM 模型名称 |
| `--group-id ID` | 存储 group ID（路径前缀） |
| `--storage-backend TYPE` | 存储后端类型: `local` / `s3` / `oss` |
| `--storage-endpoint URL` | 存储 endpoint |
| `--storage-bucket NAME` | 存储 bucket 名称 |
| `--local-root PATH` | 本地存储根目录 |
| `--use-skillclaw-config` | 从 SkillClaw 主配置加载存储/LLM 设置 |
| `-v, --verbose` | 输出 DEBUG 级别日志 |

## 环境变量

### 存储配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EVOLVE_STORAGE_BACKEND` | 后端类型 (`local` / `s3` / `oss`) | 自动推断 |
| `EVOLVE_STORAGE_ENDPOINT` | 存储 endpoint | — |
| `EVOLVE_STORAGE_BUCKET` | 存储 bucket 名称 | — |
| `EVOLVE_STORAGE_ACCESS_KEY_ID` | Access Key ID | — |
| `EVOLVE_STORAGE_SECRET_ACCESS_KEY` | Access Key Secret | — |
| `EVOLVE_STORAGE_LOCAL_ROOT` | 本地后端根目录 | — |
| `EVOLVE_GROUP_ID` | 分组 ID（存储路径前缀） | `default` |

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | LLM API 密钥 | — |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.openai.com/v1` |
| `EVOLVE_MODEL` | LLM 模型 | `gpt-4o` |
| `EVOLVE_LLM_MAX_TOKENS` | 最大生成 token 数 | `4096` |
| `EVOLVE_LLM_TEMPERATURE` | 采样温度 | `0.4` |

### 调度

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EVOLVE_INTERVAL` | 定时间隔（秒） | `600` |
| `EVOLVE_PORT` | HTTP 端口 | `8787` |
| `EVOLVE_HISTORY_LOG` | 进化历史日志路径 | `evolve_history.jsonl` |

## 输出产物

共享存储：

```
{group_id}/
├── sessions/                             ← 队列缓存（进化成功后被清空）
├── skills/{skill-name}/SKILL.md          ← 进化后的 skill 文件
├── manifest.jsonl                        ← skill 元数据清单
└── evolve_skill_registry.json            ← skill ID + 版本注册表
```

本地：

- `evolve_history.jsonl` — 每轮进化的完整 summary 记录

## 进化记录结构

每条进化记录包含：

| 字段 | 说明 |
|------|------|
| `action` | 实际执行的动作（improve_skill / optimize_description / create_skill / merge） |
| `skill_name` | 进化的 skill 名称 |
| `skill_id` | 确定性 ID (SHA-256[:12]) |
| `version` | 进化后的版本号 |
| `session_ids` | 关联的 session ID 列表 |
| `rationale` | LLM 给出的进化理由 |
| `source` | 来源 (`skill_group` / `no_skill`) |
| `edit_summary` | LLM 生成的编辑摘要 |
