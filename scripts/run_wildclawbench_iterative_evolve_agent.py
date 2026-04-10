#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve_server.config import AgentEvolveServerConfig
from agent_evolve_server.server import AgentEvolveServer
from skillclaw.experiments.group_benchmark import GroupBenchmarkConfig, GroupBenchmarkRunner
from skillclaw.experiments.wildclawbench_batch_evolve_experiment import (
    WildClawBenchBatchEvolveExperimentConfig,
)
from skillclaw.experiments.wildclawbench_iterative_evolve import IterativeEvolveRunner


def _first_nonempty(*values: str, default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return default


def _resolve_openclaw_bin(config_value: str) -> str:
    """Prefer an explicit override; otherwise keep bare executable names bare.

    The shared YAML loader resolves path-like config fields relative to the
    config file, which turns the default ``openclaw`` executable into a
    non-existent path such as ``.../experiments/.../openclaw``. For the
    host-side agent evolve step we want the shell-resolved executable unless a
    real path was intentionally provided.
    """
    override = _env("AGENT_EVOLVE_OPENCLAW_BIN")
    if override:
        return override

    value = str(config_value or "").strip()
    if not value:
        return "openclaw"

    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    if path.name == "openclaw":
        return "openclaw"
    return value


def _fresh_mode() -> str:
    raw = _env("AGENT_EVOLVE_FRESH_MODE", default="first-round").lower()
    if raw in {"always", "never", "first-round"}:
        return raw
    return "first-round"


def _fresh_for_round(round_idx: int) -> bool:
    mode = _fresh_mode()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return round_idx == 1


class AgentIterativeEvolveRunner(IterativeEvolveRunner):
    """Iterative runner that swaps classic evolve_server for agent_evolve_server."""

    def _run_round_evolution(
        self,
        *,
        evolve_arm_root: Path,
        round_idx: int,
        train_summary: dict[str, Any],
        prev_cloud_dir: Path | None,
        prev_skills_dir: Path,
        task_specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evolve_arm_root.mkdir(parents=True, exist_ok=True)
        cloud_dir = evolve_arm_root / "cloud"
        cloud_dir.mkdir(parents=True, exist_ok=True)

        if prev_cloud_dir is not None and prev_cloud_dir.exists():
            import shutil

            shutil.copytree(prev_cloud_dir, cloud_dir, dirs_exist_ok=True)

        group_id = f"{self.config.name}-iterative-round-{round_idx}"

        if prev_skills_dir.exists():
            from skillclaw.experiments.local_skill_hub import LocalSkillHub

            hub = LocalSkillHub(
                root_dir=str(cloud_dir),
                group_id=group_id,
                user_alias="seed",
            )
            hub.push_skills(str(prev_skills_dir))

        rollouts = max(1, self.config.after_rollouts_per_task)
        arm_devices = self.inner._arm_device_count(task_specs, rollouts)
        debug_dir = (evolve_arm_root / "evolve_debug").resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)
        runner = GroupBenchmarkRunner(
            GroupBenchmarkConfig(
                name=f"{self.config.name}-iterative-round-{round_idx}",
                workspace_dir=str(evolve_arm_root / "workspace"),
                tasks_path=str(evolve_arm_root / "tasks.jsonl"),
                cloud_dir=str(cloud_dir),
                group_id=group_id,
                devices=arm_devices,
                cluster_configured_nodes=max(
                    self.config.cluster_configured_nodes,
                    arm_devices,
                ),
                cluster_llm_api_base=self.config.cluster_llm_api_base,
                cluster_llm_api_key=self.config.cluster_llm_api_key,
                cluster_llm_model_id=self.config.cluster_llm_model_id,
                evolve_api_base=self.config.evolve_api_base or self.config.cluster_llm_api_base,
                evolve_api_key=self.config.evolve_api_key or self.config.cluster_llm_api_key,
                evolve_model=self.config.evolve_model or self.config.cluster_llm_model_id,
                evolve_max_tokens=self.config.evolve_max_tokens,
                evolve_debug_dir=str(debug_dir),
                evolve_use_success_feedback=self.config.evolve_use_success_feedback,
                # Agent-evolve should inspect the original session files directly
                # instead of collapsing them into per-task synthetic sessions.
                evolve_aggregate_sessions=False,
                executor="replay",
            )
        )

        aggregate_info = runner._prepare_evolution_backend(
            round_idx,
            train_summary=train_summary,
        )

        run_root = evolve_arm_root.parent.parent.parent
        shared_state_root = run_root / "agent_evolve_state"
        shared_state_root.mkdir(parents=True, exist_ok=True)

        llm_api_key = _env(
            "AGENT_EVOLVE_LLM_API_KEY",
            "AGENT_EVOLVE_API_KEY",
            "SKILLCLAW_EVOLVE_API_KEY",
            "SKILLCLAW_LLM_API_KEY",
            "OPENAI_API_KEY",
            default=self.config.evolve_api_key or self.config.cluster_llm_api_key,
        )
        llm_base_url = _env(
            "AGENT_EVOLVE_LLM_BASE_URL",
            "AGENT_EVOLVE_BASE_URL",
            default=self.config.evolve_api_base or self.config.cluster_llm_api_base,
        )
        llm_model = _env(
            "AGENT_EVOLVE_MODEL",
            "SKILLCLAW_EVOLVE_MODEL",
            "SKILLCLAW_LLM_MODEL",
            default=self.config.evolve_model or self.config.cluster_llm_model_id,
        )
        llm_api_type = _env(
            "AGENT_EVOLVE_LLM_API_TYPE",
            default="openai-completions",
        )
        agent_timeout = int(
            _env(
                "AGENT_EVOLVE_TIMEOUT",
                default=str(max(900, self.config.cluster_task_timeout_seconds + 180)),
            )
        )
        fresh = _fresh_for_round(round_idx)

        config = AgentEvolveServerConfig(
            storage_backend="local",
            group_id=group_id,
            local_root=str(aggregate_info["backend_root"]),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_max_tokens=self.config.evolve_max_tokens,
            openclaw_bin=_resolve_openclaw_bin(self.config.cluster_openclaw_bin),
            openclaw_home=str(shared_state_root / "openclaw_home"),
            fresh=fresh,
            agent_timeout=agent_timeout,
            workspace_root=str(shared_state_root / "workspace"),
            llm_api_type=llm_api_type,
            history_path=str(Path(aggregate_info["round_dir"]) / "agent_evolve_history.jsonl"),
        )

        summary = asyncio.run(AgentEvolveServer(config).run_once())
        runner._sync_evolution_backend_to_cloud(Path(aggregate_info["backend_root"]))
        return {
            "round": round_idx,
            "summary": summary,
            "manifest_path": str(cloud_dir / group_id / "manifest.jsonl"),
            "registry_path": str(cloud_dir / group_id / "evolve_skill_registry.json"),
            "aggregation": {
                "enabled": runner.config.evolve_aggregate_sessions,
                "backend_root": str(aggregate_info["backend_root"]),
                "source_session_count": aggregate_info["source_session_count"],
                "aggregated_session_count": aggregate_info["aggregated_session_count"],
                "aggregated_session_paths": aggregate_info["aggregated_session_paths"],
            },
            "agent_evolve": {
                "fresh": fresh,
                "fresh_mode": _fresh_mode(),
                "llm_api_type": llm_api_type,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "workspace_root": config.workspace_root,
                "openclaw_home": config.openclaw_home,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run WildClawBench iterative evolve using agent_evolve_server.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to WildClawBench iterative evolve YAML/JSON config.",
    )
    args = parser.parse_args()

    config = WildClawBenchBatchEvolveExperimentConfig.from_file(args.config)
    report = AgentIterativeEvolveRunner(config).run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
