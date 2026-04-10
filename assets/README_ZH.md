<div align="center">

<img src="./skillclaw_logo.png" alt="SkillClaw" width="150">

# SkillClaw

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-Visit%20Site-green?style=flat-square&logo=homepage)](https://your-homepage-url.com)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.08377)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobeacrobatreader)](https://arxiv.org/pdf/2604.08377)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/papers/2604.08377)

</div>

</div>

> [!NOTE]
> SkillClaw 是一个面向多用户 OpenClaw 生态的**技能群体进化框架**。把多个用户和多个 Agent 的实战经验自动提炼为可复用的 Skill，通过云端共享让整个 Agent 集群持续进化。

<div align="center">

<img src="./skillclaw_main.png" alt="SkillClaw 架构图" width="800">

[English](../README.md)

</div>

**安装极简，使用无感** —— 只需两行命令即可完成安装，支持各种场景。对于用户来说，只需要正常和 OpenClaw 对话，Skill 的自动进化在后台无感完成，无需任何额外操作。

**广泛兼容** —— SkillClaw 无缝支持各种 Claw 框架，包括 CoPaw、IronClaw、PicoClaw、ZeroClaw、NanoClaw、NemoClaw。

**效果显著** —— 在 [WildClawBench](https://github.com/InternLM/WildClawBench) 的真实世界场景评测中，即使在有限的群体交互和反馈条件下，SkillClaw 也显著提升了 Qwen3-Max 的表现——不靠更大的模型，靠更聪明的经验。

---

## 项目概览

SkillClaw 通过从真实会话数据中**演化可复用技能**并在一组 Agent 之间共享，让 LLM Agent 持续变强。

系统由三个部分组成：

1. **Client Proxy**：本地 API 代理（`/v1/chat/completions`、`/v1/messages`），负责拦截 Agent 请求、记录会话产物，并与共享存储同步技能。
2. **Workflow Evolve Server**（`evolve_server`）：固定三阶段 LLM 流程（Summarize -> Aggregate -> Execute），从共享存储读取会话数据，演化或创建技能，再写回存储。
3. **Agent Evolve Server**（`agent_evolve_server`）：工作流版 evolve server 的自主 Agent 替代方案。它使用 OpenClaw agent 读取会话、分析模式，并直接写入演化后的技能文件，具备完整读写执行能力。

三者共享同一套存储层（Alibaba OSS / S3 / 本地文件系统）以及同一种技能格式（`SKILL.md`），因此可以在同一部署体系中互相替换。

---

## 快速开始

### 前置要求

- Python >= 3.10
- 一个兼容 OpenAI 的 LLM API endpoint

### 1. 安装

二选一：

```bash
# Client / 本地开发
git clone <repo-url> SkillClaw && cd SkillClaw
bash scripts/install_skillclaw.sh
source .venv/bin/activate

# Server 部署（同时安装两个 server 入口）
bash scripts/install_skillclaw_server.sh
source .venv-server/bin/activate

# 仅在使用 agent evolve server 时需要
npm install -g openclaw
```

### 2. 配置并启动 Client Proxy

```bash
export OPENAI_BASE_URL="https://your-api-gateway/v1"
export OPENAI_API_KEY="sk-..."

skillclaw setup
skillclaw start
skillclaw status
skillclaw config show
```

### 3. 启动 Evolve Server

SkillClaw 提供两种 evolve server，可任选其一：

Workflow evolve server：

```bash
skillclaw-evolve-server --port 8787 --interval 300 \
  --storage-backend oss \
  --oss-endpoint "$EVOLVE_STORAGE_ENDPOINT" \
  --oss-bucket "$EVOLVE_STORAGE_BUCKET" \
  --group-id my-group
```

Agent evolve server：

```bash
skillclaw-agent-evolve-server --port 8787 --interval 300 --no-fresh \
  --storage-backend oss \
  --oss-endpoint "$EVOLVE_STORAGE_ENDPOINT" \
  --oss-bucket "$EVOLVE_STORAGE_BUCKET" \
  --group-id my-group
```

> `skillclaw-agent-evolve-server` 会由 `scripts/install_skillclaw_server.sh` 一并安装，但仍然要求 `openclaw` 在 PATH 中可用。模型和运行时细节见 [`agent_evolve_server/README.md`](../agent_evolve_server/README.md)。

### 4. 技能管理

```bash
skillclaw skills pull          # 下载共享技能
skillclaw skills push          # 上传本地技能
skillclaw skills sync          # 双向同步
skillclaw skills list-remote   # 查看远端技能
```

---

## WildClawBench 实验

SkillClaw 内置了用于 [WildClawBench](https://github.com/InternLM/WildClawBench) 的技能演化实验流程。当前对外保留的主要实验入口是 `scripts/run_wildclawbench_iterative_evolve_agent.py`，其余实验能力可通过 `skillclaw benchmark` CLI 使用。完整子命令请查看 `skillclaw benchmark --help`。

---

## 项目结构

```text
SkillClaw/
├── skillclaw/                  # Client proxy、CLI、配置、技能同步、实验逻辑
│   ├── cli.py
│   ├── api_server.py
│   ├── launcher.py
│   ├── skill_manager.py / skill_hub.py
│   └── experiments/
├── evolve_server/              # 工作流式 evolve server
│   ├── __main__.py
│   ├── server.py
│   ├── summarizer.py / aggregation.py / execution.py
│   └── config.py / llm_client.py / skill_registry.py
├── agent_evolve_server/        # 基于 OpenClaw 的 evolve server
│   ├── __main__.py
│   ├── server.py
│   ├── workspace.py / openclaw_runner.py
│   └── EVOLVE_AGENTS.md
├── scripts/                    # 安装脚本 + 主要公开实验入口
│   ├── install_skillclaw.sh
│   ├── install_skillclaw_server.sh
│   └── run_wildclawbench_iterative_evolve_agent.py
├── assets/                     # Logo 与文档资源
└── pyproject.toml              # 包元数据与 extras
```

## 配置

配置由 `~/.skillclaw/config.yaml` 和环境变量共同组成。

- Client / 共享凭证模板：[`example_env.sh`](../example_env.sh)
- Evolve server 环境变量模板：[`evolve_server/.env.example`](../evolve_server/.env.example)
- 查看配置：`skillclaw config show`
- 修改配置：`skillclaw config <key> <value>`

## 致谢

本仓库基于以下开源项目构建：

- [MetaClaw](https://github.com/aiming-lab/MetaClaw) - Just talk to your agent — it learns and evolves
- [WildClawBench](https://github.com/InternLM/WildClawBench) - Can an AI agent do real work, end-to-end, without hand-holding
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) - Train a personalized agent simply by talking to it

## 共建

SkillClaw 是一个社区共建项目。我们欢迎各种形式的贡献——Bug 反馈、功能建议、新技能贡献、文档完善等。欢迎提 Issue 或提交 Pull Request！

## 许可证

详见 [LICENSE](../LICENSE)。
