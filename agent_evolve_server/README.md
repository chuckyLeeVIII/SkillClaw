# agent_evolve_server

Agent-driven skill evolution server for SkillClaw. Replaces the fixed 3-stage LLM pipeline in `evolve_server` with an autonomous OpenClaw agent that reads session data, analyzes patterns, and writes evolved SKILL.md files.

Default model: **claude-opus-4-6** via Anthropic Messages API, using the local gateway at `http://localhost:28080` unless overridden.

## vs `evolve_server`

Both servers share the same external interface (`run_once`, `run_periodic`, HTTP trigger) and the same storage layer (OSS / S3 / local), but differ in **how** evolution happens:

| | `evolve_server` | `agent_evolve_server` |
|---|---|---|
| Evolution engine | Fixed 3-stage LLM pipeline: Summarize → Aggregate → Execute | Session summarizer + autonomous OpenClaw agent with tools (read/write/exec) |
| LLM calls | Multiple structured calls per stage, each with a fixed prompt template | Single agent session; the agent decides what to read, analyze, and write |
| Default model | Configurable (typically `gpt-5.4`) | `claude-opus-4-6` via `anthropic-messages` |
| Skill editing | LLM generates diffs via structured output; server applies them | Agent directly writes/edits SKILL.md files in the workspace |
| History tracking | Server-managed versioning | Agent-managed `history/` directories per skill, guided by EVOLVE_AGENTS.md |
| State across rounds | Stateless (each cycle is independent) | `--fresh` controls: stateless or persistent agent memory across rounds |
| Configuration | `EvolveServerConfig` | `AgentEvolveServerConfig` (extends the former with OpenClaw settings) |
| Dependencies | Only needs LLM API | Needs LLM API + OpenClaw binary installed |

### Architecture

```
Storage (OSS/S3/local)
    │
    ▼
AgentEvolveServer.run_once()
    │
    ├─ 1. Drain sessions from storage
    ├─ 1.5 Summarize sessions (`_trajectory` + `_summary` + metadata)
    ├─ 2. Prepare workspace with compact pre-processed session JSON
    ├─ 3. Snapshot skill hashes
    ├─ 4. Run OpenClaw agent subprocess
    │      └─ Agent reads EVOLVE_AGENTS.md → analyzes sessions → writes skills
    ├─ 5. Collect changes (hash diff)
    ├─ 6. Upload changed skills to storage
    └─ 7. Ack (delete consumed sessions)
```

The workspace is populated with custom bootstrap files (`AGENTS.md`, `SOUL.md`, etc.) that direct the agent to focus exclusively on skill evolution. Session files written into `workspace/sessions/` are compact, pre-processed JSON records containing metadata, aggregate stats, `_trajectory`, and `_summary` rather than the original raw `turns` arrays. The agent follows the methodology defined in `EVOLVE_AGENTS.md`.

## Quick Start

### Prerequisites

- Python >= 3.10 with SkillClaw's server dependencies installed
- OpenClaw installed and available in PATH (`npm install -g openclaw`)
- LLM API credentials for your chosen provider (`anthropic-messages` by default; OpenAI-compatible endpoints also supported)

Recommended install from the repository root:

```bash
bash scripts/install_skillclaw_server.sh
source .venv-server/bin/activate
npm install -g openclaw
```

### Single run (local storage)

```bash
cd /path/to/SkillClaw

python -m agent_evolve_server \
  --once \
  --storage-backend local \
  --local-root /path/to/storage \
  --group-id my-experiment \
  --fresh \
  -v
```

This uses the default `claude-opus-4-6` model. Sessions are read from `/path/to/storage/my-experiment/sessions/`, summarized into compact workspace files, and evolved skills are written back to `/path/to/storage/my-experiment/skills/`.

### Using a different model

Override model and API type via CLI:

```bash
# Use gpt-5.4 via OpenAI-compatible endpoint
python -m agent_evolve_server --once --fresh \
  --model gpt-5.4 \
  --llm-api-type openai-completions \
  ...

# Use Claude Sonnet via Anthropic Messages API
python -m agent_evolve_server --once --fresh \
  --model claude-sonnet-4-6 \
  ...
```

Or via environment variables (useful in `.env`):

```bash
export AGENT_EVOLVE_MODEL=claude-opus-4-6
export AGENT_EVOLVE_LLM_BASE_URL=http://localhost:28080
export AGENT_EVOLVE_LLM_API_TYPE=openai-completions   # default: anthropic-messages
```

### Periodic mode (remote storage)

```bash
python -m agent_evolve_server \
  --storage-backend oss \
  --oss-endpoint $EVOLVE_STORAGE_ENDPOINT \
  --oss-bucket $EVOLVE_STORAGE_BUCKET \
  --group-id my-experiment \
  --interval 300 \
  --no-fresh \
  -v
```

Polls storage every 300 seconds. Uses `--no-fresh` to preserve agent memory across rounds.

### With HTTP trigger

```bash
python -m agent_evolve_server \
  --port 8787 \
  --storage-backend local \
  --local-root /path/to/storage \
  --group-id my-experiment \
  --no-fresh
```

Exposes `/trigger` (POST), `/status` (GET), `/health` (GET).

## `--fresh` vs `--no-fresh`: Multi-Round Evolution

This is the most important behavioral switch:

### `--fresh` (default)

Each evolution cycle is **fully independent**:

- The agent workspace is wiped before every cycle
- OpenClaw home (agent state, session history) is wiped
- No memory carries over between rounds
- The agent sees only the current sessions and current skills

Use this for: **the first round of a new category**, isolated single-round experiments, debugging, or when you explicitly want no state leakage between cycles. When switching to a different task category, always start with `--fresh`.

```bash
python -m agent_evolve_server --once --fresh ...
```

### `--no-fresh`

Cycles are **cumulative** — the agent retains context across rounds:

- The workspace is NOT wiped; `memory/` and `MEMORY.md` persist
- OpenClaw home preserves the agent's session history
- The agent can read its own notes from previous rounds
- Skill `history/` directories accumulate across rounds (these persist regardless of `--fresh` since they live in storage)

Use this for: **round 2+ of the same category** in multi-round iterative evolution, where the agent should learn from its own past decisions and avoid reverting previous improvements. **Do NOT use `--fresh` between rounds of the same category** — it would erase the agent's memory and force it to re-derive or contradict earlier evolution decisions.

```bash
# Round 1: first batch of sessions arrives
python -m agent_evolve_server --once --no-fresh \
  --storage-backend local --local-root ./storage --group-id exp1 ...

# Round 2: new sessions arrive, agent remembers round 1
python -m agent_evolve_server --once --no-fresh \
  --storage-backend local --local-root ./storage --group-id exp1 ...

# Round N: agent has full history context
python -m agent_evolve_server --once --no-fresh \
  --storage-backend local --local-root ./storage --group-id exp1 ...
```

In periodic mode, `--no-fresh` is the natural choice:

```bash
python -m agent_evolve_server --interval 300 --no-fresh ...
```

### Experiment best practice: when to use which

In a typical WildClawBench experiment, each **category** (e.g. `02_Code_Intelligence`) runs multiple rounds of evaluation + evolution. The rule of thumb:

| Scenario | Flag | Why |
|---|---|---|
| **Same category, round N → round N+1** | `--no-fresh` | The agent should retain memory from previous rounds — it already knows which skills it created/updated and why. Wiping state would cause it to re-derive the same conclusions or, worse, contradict its earlier decisions. |
| **Switching to a new category** | `--fresh` | A new category has entirely different tasks and session patterns. Previous memory is irrelevant and could mislead the agent. Start clean. |
| **Debugging / one-off test** | `--fresh` | Ensures reproducibility with no leftover state. |

Example: running `02_Code_Intelligence` across 3 rounds, then switching to `04_Search_Retrieval`:

```bash
# Category: 02_Code_Intelligence
# Round 1 (baseline → first evolution): fresh start
python -m agent_evolve_server --once --fresh \
  --group-id 02-code-intel ...

# Round 2: new sessions arrive, agent remembers round 1
python -m agent_evolve_server --once --no-fresh \
  --group-id 02-code-intel ...

# Round 3: agent continues building on rounds 1+2
python -m agent_evolve_server --once --no-fresh \
  --group-id 02-code-intel ...

# Switch to a new category: wipe everything
python -m agent_evolve_server --once --fresh \
  --group-id 04-search-retrieval ...
```

> **Key point**: within the same category, always use `--no-fresh` after the first round. Only use `--fresh` when starting a brand-new category or when you explicitly want a clean-slate experiment.

### What persists where

| Content | `--fresh` | `--no-fresh` |
|---|---|---|
| `workspace/sessions/` | Cleared each cycle | Cleared each cycle |
| `workspace/skills/` | Re-populated from storage | Re-populated from storage |
| `workspace/memory/`, `MEMORY.md` | **Wiped** | **Preserved** |
| `workspace/AGENTS.md`, `SOUL.md`, ... | Re-written each cycle | Re-written each cycle |
| `skills/<name>/history/` in storage | Persists (in storage) | Persists (in storage) |
| OpenClaw home (`.openclaw_home/`) | **Wiped** | **Preserved** |

## CLI Reference

```
python -m agent_evolve_server [OPTIONS]
```

### Execution modes

| Flag | Description |
|---|---|
| `--once` | Run one cycle and exit |
| `--mock` | Use local `mock/` directory (for development) |
| `--mock-root PATH` | Custom mock directory |
| `--port PORT` | Enable HTTP server on PORT |
| `--interval SECONDS` | Periodic polling interval |

### Storage

| Flag | Description |
|---|---|
| `--storage-backend {local,oss,s3}` | Storage backend type |
| `--local-root PATH` | Root directory for local backend |
| `--group-id ID` | Shared storage group prefix |
| `--oss-endpoint URL` | Alibaba Cloud OSS endpoint |
| `--oss-bucket NAME` | OSS bucket name |
| `--storage-endpoint URL` | S3-compatible endpoint |
| `--storage-bucket NAME` | S3 bucket name |
| `--storage-region REGION` | S3 region |
| `--use-skillclaw-config` | Load settings from SkillClaw config store |

### Agent & Model

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | `claude-opus-4-6` | LLM model name |
| `--llm-api-type TYPE` | `anthropic-messages` | LLM provider API type |
| `--fresh` / `--no-fresh` | `--fresh` | Wipe or preserve agent state between cycles |
| `--agent-timeout SECONDS` | `600` | Agent execution timeout |
| `--workspace-root PATH` | `agent_evolve_server/agent_workspace` | Agent workspace directory |
| `--agents-md PATH` | built-in `EVOLVE_AGENTS.md` | Custom EVOLVE_AGENTS.md path |
| `--openclaw-bin PATH` | `openclaw` | OpenClaw binary path |
| `--openclaw-home PATH` | `agent_evolve_server/.openclaw_home` | OpenClaw home directory |

Supported `--llm-api-type` values:

| Value | Protocol | Typical providers |
|---|---|---|
| `anthropic-messages` | Anthropic `/v1/messages` | Anthropic API, compatible proxies |
| `openai-completions` | OpenAI `/v1/chat/completions` | OpenAI, Azure, OpenRouter, vLLM |
| `openai-responses` | OpenAI Responses API | OpenAI (newer) |
| `google-generative-ai` | Google Generative AI | Gemini |
| `ollama` | Ollama native | Local Ollama |

### Environment variables

All CLI flags can also be set via environment variables:

| Variable | Default | Maps to |
|---|---|---|
| `AGENT_EVOLVE_LLM_API_KEY` | — | Agent-evolve LLM API key |
| `AGENT_EVOLVE_API_KEY` | — | Legacy alias for `AGENT_EVOLVE_LLM_API_KEY` |
| `AGENT_EVOLVE_LLM_BASE_URL` | `http://localhost:28080` | Agent-evolve LLM API base URL |
| `AGENT_EVOLVE_BASE_URL` | — | Legacy alias for `AGENT_EVOLVE_LLM_BASE_URL` |
| `AGENT_EVOLVE_MODEL` | `claude-opus-4-6` | Agent-evolve LLM model |
| `AGENT_EVOLVE_LLM_API_TYPE` | `anthropic-messages` | `--llm-api-type` |
| `AGENT_EVOLVE_FRESH` | `1` (true) | `--fresh` / `--no-fresh` |
| `AGENT_EVOLVE_TIMEOUT` | `600` | `--agent-timeout` |
| `AGENT_EVOLVE_OPENCLAW_BIN` | `openclaw` | `--openclaw-bin` |
| `AGENT_EVOLVE_OPENCLAW_HOME` | — | `--openclaw-home` |
| `AGENT_EVOLVE_WORKSPACE_ROOT` | — | `--workspace-root` |
| `AGENT_EVOLVE_AGENTS_MD` | — | `--agents-md` |
| `OPENAI_API_KEY` | — | Fallback API key if no agent-specific key is set |
| `EVOLVE_STORAGE_*` | — | Storage credentials (inherited from `evolve_server`) |

## Files

```
agent_evolve_server/
├── __init__.py          # Public API: AgentEvolveServer, AgentEvolveServerConfig
├── __main__.py          # CLI entry point
├── config.py            # AgentEvolveServerConfig (extends EvolveServerConfig)
├── server.py            # AgentEvolveServer — main orchestrator
├── workspace.py         # AgentWorkspace — file preparation & change collection
├── openclaw_runner.py   # OpenClawRunner — subprocess management
├── agents_md.py         # Built-in EVOLVE_AGENTS.md loader
└── EVOLVE_AGENTS.md     # Evolution methodology (injected into workspace)
```
