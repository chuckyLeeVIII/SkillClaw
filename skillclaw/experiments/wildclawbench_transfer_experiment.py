"""
Two-arm WildClawBench transfer experiment runner.

This experiment compares:

1. baseline: all devices directly evaluate the same task set
2. evolve_then_broadcast: one train task per device -> evolve -> broadcast -> full eval

It intentionally reuses the existing group benchmark + WildClawBench executor
so the task execution path stays identical to the rest of the repo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .group_benchmark import run_benchmark_from_config
from .wildclawbench_executor import parse_wildclaw_task_md


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return 0


def _load_manifest_names(path: str) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    names: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = str(payload.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _flatten_phase_records(phase_summary: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for device in phase_summary.get("devices", []) or []:
        records.extend(device.get("records", []) or [])
    return records


def _aggregate_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "elapsed_time_s": 0.0,
        "task_count": len(records),
    }
    for record in records:
        wildbench = ((record.get("artifacts") or {}).get("wildclawbench") or {})
        usage = wildbench.get("usage") or {}
        totals["input_tokens"] += _safe_int(usage.get("input_tokens"))
        totals["output_tokens"] += _safe_int(usage.get("output_tokens"))
        totals["cache_read_tokens"] += _safe_int(usage.get("cache_read_tokens"))
        totals["cache_write_tokens"] += _safe_int(usage.get("cache_write_tokens"))
        totals["total_tokens"] += _safe_int(usage.get("total_tokens"))
        totals["cost_usd"] += _safe_float(usage.get("cost_usd"))
        totals["elapsed_time_s"] += _safe_float(usage.get("elapsed_time"))
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["elapsed_time_s"] = round(totals["elapsed_time_s"], 3)
    return totals


@dataclass
class WildClawBenchTransferExperimentConfig:
    name: str = "wildclawbench-transfer"
    workspace_dir: str = "records/wildclawbench_transfer_experiment"
    benchmark_root: str = ""
    task_files: list[str] = field(default_factory=list)
    eval_task_files: list[str] = field(default_factory=list)
    group_id: str = "wildbench-transfer"
    initial_skills_dir: str = ""
    devices: int = 8
    baseline_devices: int = 1
    post_eval_devices: int = 1
    max_parallel_devices: int = 8
    cluster_configured_nodes: int = 8
    cluster_skillclaw_base_port: int = 39000
    cluster_gateway_base_port: int = 41000
    cluster_openclaw_bin: str = "openclaw"
    cluster_skillclaw_bin: str = "skillclaw"
    cluster_task_timeout_seconds: int = 180
    cluster_start_timeout_seconds: float = 20.0
    cluster_openclaw_mode: str = "gateway"
    cluster_seed_openclaw_dir: str = ""
    cluster_llm_provider: str = "custom"
    cluster_llm_api_base: str = ""
    cluster_llm_api_key: str = ""
    cluster_llm_model_id: str = "gpt-5.4"
    cluster_proxy_api_key: str = "skillclaw"
    cluster_retrieval_mode: str = "template"
    cluster_prm_enabled: bool = False
    cluster_prm_provider: str = "openai"
    cluster_prm_url: str = "https://api.openai.com/v1"
    cluster_prm_model: str = "gpt-5.2"
    cluster_prm_api_key: str = ""
    evolve_api_base: str = ""
    evolve_api_key: str = ""
    evolve_model: str = ""
    evolve_max_tokens: int = 100000
    evolve_strategy: str = "dynamic_edit_conservative"
    evolve_use_success_feedback: bool = True
    eval_assignment_strategy: str = "replicate"
    wildclawbench_gateway_port: int = 18789
    wildclawbench_container_proxy_host: str = "host.docker.internal"
    wildclawbench_docker_image: str = ""
    wildclawbench_tmp_workspace: str = "/tmp_workspace"
    wildclawbench_task_timeout_seconds_override: int = 0
    wildclawbench_success_threshold: float = 0.5
    wildclawbench_use_score_as_prm: bool = True
    strict_require_nonempty_response: bool = False
    strict_require_conversation_records: bool = False
    strict_require_skill_reads_from_local_dir: bool = False
    strict_forbidden_substrings: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "WildClawBenchTransferExperimentConfig":
        cfg_path = Path(path).expanduser().resolve()
        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("wildclawbench transfer config must deserialize to a dict")

        base_dir = cfg_path.parent

        def _expand_env(obj: Any) -> Any:
            if isinstance(obj, str):
                return _expand_string_env(obj)
            if isinstance(obj, list):
                return [_expand_env(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _expand_env(value) for key, value in obj.items()}
            return obj

        data = _expand_env(data)

        for key in (
            "workspace_dir",
            "benchmark_root",
            "initial_skills_dir",
            "cluster_seed_openclaw_dir",
            "cluster_openclaw_bin",
            "cluster_skillclaw_bin",
        ):
            value = data.get(key)
            if isinstance(value, str) and value:
                value = _expand_string_env(value)
                if not Path(value).expanduser().is_absolute():
                    data[key] = str((base_dir / value).resolve())
                else:
                    data[key] = str(Path(value).expanduser().resolve())

        benchmark_root_value = str(data.get("benchmark_root") or "").strip()
        benchmark_root = Path(benchmark_root_value).expanduser() if benchmark_root_value else None

        def _resolve_task_list(items: list[Any]) -> list[str]:
            resolved: list[str] = []
            for raw_task in items or []:
                value = _expand_string_env(str(raw_task))
                path_obj = Path(value).expanduser()
                if path_obj.is_absolute():
                    resolved.append(str(path_obj.resolve()))
                elif benchmark_root is not None:
                    resolved.append(str((benchmark_root / path_obj).resolve()))
                else:
                    resolved.append(str((base_dir / path_obj).resolve()))
            return resolved

        data["task_files"] = _resolve_task_list(data.get("task_files", []) or [])
        data["eval_task_files"] = _resolve_task_list(data.get("eval_task_files", []) or [])

        for key in (
            "cluster_llm_api_base",
            "cluster_llm_api_key",
            "cluster_llm_model_id",
            "evolve_api_base",
            "evolve_api_key",
            "evolve_model",
        ):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = _expand_string_env(value)

        return cls(**data)


def _expand_string_env(value: str) -> str:
    import os

    return os.path.expandvars(value)


class WildClawBenchTransferExperimentRunner:
    def __init__(self, config: WildClawBenchTransferExperimentConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir).expanduser().resolve()
        self.benchmark_root = Path(config.benchmark_root).expanduser().resolve()

    def run(self) -> dict[str, Any]:
        train_task_specs = self._load_task_specs(self.config.task_files)
        eval_task_specs = self._load_task_specs(self.config.eval_task_files or self.config.task_files)
        if len(train_task_specs) != self.config.devices:
            raise ValueError(
                f"transfer experiment requires devices == number of task_files; got devices={self.config.devices}, tasks={len(train_task_specs)}"
            )

        run_root = self.workspace_dir / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        initial_skills_dir = self._ensure_initial_skills_dir(run_root)

        baseline_report = self._run_baseline_arm(run_root, eval_task_specs, initial_skills_dir)
        evolve_report = self._run_evolve_then_broadcast_arm(
            run_root,
            train_task_specs,
            eval_task_specs,
            initial_skills_dir,
        )

        report = self._build_report(
            run_root=run_root,
            train_task_specs=train_task_specs,
            eval_task_specs=eval_task_specs,
            baseline_report=baseline_report,
            evolve_report=evolve_report,
        )
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "report.md").write_text(self._build_markdown(report), encoding="utf-8")
        return report

    def _ensure_initial_skills_dir(self, run_root: Path) -> Path:
        if self.config.initial_skills_dir:
            path = Path(self.config.initial_skills_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        path = run_root / "initial_skills"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_task_specs(self, task_files: list[str]) -> list[dict[str, str]]:
        if not self.benchmark_root.exists():
            raise FileNotFoundError(f"benchmark_root not found: {self.benchmark_root}")
        specs: list[dict[str, str]] = []
        for raw_path in task_files:
            task_path = Path(raw_path).expanduser().resolve()
            if not task_path.exists():
                raise FileNotFoundError(f"task_file not found: {task_path}")
            parsed = parse_wildclaw_task_md(task_path, benchmark_root=self.benchmark_root)
            specs.append(
                {
                    "task_id": parsed.task_id,
                    "task_file": parsed.file_path,
                    "category": parsed.category,
                }
            )
        return specs

    def _run_baseline_arm(
        self,
        run_root: Path,
        task_specs: list[dict[str, str]],
        initial_skills_dir: Path,
    ) -> dict[str, Any]:
        arm_root = run_root / "arms" / "baseline"
        tasks_path = arm_root / "tasks.jsonl"
        _append_jsonl(
            tasks_path,
            [
                {
                    "task_id": spec["task_id"],
                    "split": "eval",
                    "task_file": spec["task_file"],
                }
                for spec in task_specs
            ],
        )
        config_path = self._write_arm_config(
            arm_root=arm_root,
            arm_name="baseline",
            tasks_path=tasks_path,
            initial_skills_dir=initial_skills_dir,
            arm_devices=self.config.baseline_devices,
            rounds=0,
            sync_enabled=False,
            evolve_enabled=False,
            eval_every_round=False,
            initial_eval_enabled=True,
            eval_devices=self.config.baseline_devices,
            eval_assignment_strategy="replicate",
            device_field="",
            port_offset=0,
        )
        return self._run_benchmark(config_path)

    def _run_evolve_then_broadcast_arm(
        self,
        run_root: Path,
        train_task_specs: list[dict[str, str]],
        eval_task_specs: list[dict[str, str]],
        initial_skills_dir: Path,
    ) -> dict[str, Any]:
        arm_root = run_root / "arms" / "evolve_then_broadcast"
        tasks_path = arm_root / "tasks.jsonl"
        records = []
        for idx, spec in enumerate(train_task_specs):
            records.append(
                {
                    "task_id": spec["task_id"],
                    "split": "train",
                    "device": f"device-{idx}",
                    "task_file": spec["task_file"],
                }
            )
        for spec in eval_task_specs:
            records.append(
                {
                    "task_id": spec["task_id"],
                    "split": "eval",
                    "task_file": spec["task_file"],
                }
            )
        _append_jsonl(tasks_path, records)
        config_path = self._write_arm_config(
            arm_root=arm_root,
            arm_name="evolve_then_broadcast",
            tasks_path=tasks_path,
            initial_skills_dir=initial_skills_dir,
            arm_devices=self.config.devices,
            rounds=1,
            sync_enabled=True,
            evolve_enabled=True,
            eval_every_round=True,
            initial_eval_enabled=False,
            eval_devices=self.config.post_eval_devices,
            eval_assignment_strategy=self.config.eval_assignment_strategy,
            device_field="device",
            port_offset=200,
        )
        return self._run_benchmark(config_path)

    def _write_arm_config(
        self,
        *,
        arm_root: Path,
        arm_name: str,
        tasks_path: Path,
        initial_skills_dir: Path,
        arm_devices: int,
        rounds: int,
        sync_enabled: bool,
        evolve_enabled: bool,
        eval_every_round: bool,
        initial_eval_enabled: bool,
        eval_devices: int,
        eval_assignment_strategy: str,
        device_field: str,
        port_offset: int,
    ) -> Path:
        arm_root.mkdir(parents=True, exist_ok=True)
        max_parallel = self.config.max_parallel_devices or arm_devices
        cfg = {
            "name": f"{self.config.name}-{arm_name}",
            "workspace_dir": str(arm_root / "workspace"),
            "tasks_path": str(tasks_path),
            "cloud_dir": str(arm_root / "cloud"),
            "group_id": f"{self.config.group_id}-{arm_name}",
            "initial_skills_dir": str(initial_skills_dir),
            "devices": arm_devices,
            "eval_devices": eval_devices,
            "cluster_configured_nodes": max(self.config.cluster_configured_nodes, arm_devices),
            "rounds": rounds,
            "initial_eval_enabled": initial_eval_enabled,
            "train_tasks_per_device_per_round": 1,
            "max_parallel_devices": max_parallel,
            "sync_enabled": sync_enabled,
            "eval_every_round": eval_every_round,
            "eval_assignment_strategy": eval_assignment_strategy,
            "device_field": device_field,
            "executor": "wildclawbench_cluster",
            "wildclawbench_root": str(self.benchmark_root),
            "wildclawbench_output_dir": str(arm_root / "wildclawbench_output"),
            "wildclawbench_container_proxy_host": self.config.wildclawbench_container_proxy_host,
            "wildclawbench_gateway_port": self.config.wildclawbench_gateway_port,
            "wildclawbench_docker_image": self.config.wildclawbench_docker_image,
            "wildclawbench_tmp_workspace": self.config.wildclawbench_tmp_workspace,
            "wildclawbench_task_timeout_seconds_override": self.config.wildclawbench_task_timeout_seconds_override,
            "wildclawbench_success_threshold": self.config.wildclawbench_success_threshold,
            "wildclawbench_use_score_as_prm": self.config.wildclawbench_use_score_as_prm,
            "cluster_skillclaw_base_port": self.config.cluster_skillclaw_base_port + port_offset,
            "cluster_gateway_base_port": self.config.cluster_gateway_base_port + port_offset,
            "cluster_openclaw_bin": self.config.cluster_openclaw_bin,
            "cluster_skillclaw_bin": self.config.cluster_skillclaw_bin,
            "cluster_task_timeout_seconds": self.config.cluster_task_timeout_seconds,
            "cluster_start_timeout_seconds": self.config.cluster_start_timeout_seconds,
            "cluster_openclaw_mode": self.config.cluster_openclaw_mode,
            "cluster_seed_openclaw_dir": self.config.cluster_seed_openclaw_dir,
            "cluster_llm_provider": self.config.cluster_llm_provider,
            "cluster_llm_api_base": self.config.cluster_llm_api_base,
            "cluster_llm_api_key": self.config.cluster_llm_api_key,
            "cluster_llm_model_id": self.config.cluster_llm_model_id,
            "cluster_proxy_api_key": self.config.cluster_proxy_api_key,
            "cluster_retrieval_mode": self.config.cluster_retrieval_mode,
            "cluster_prm_enabled": self.config.cluster_prm_enabled,
            "cluster_prm_provider": self.config.cluster_prm_provider,
            "cluster_prm_url": self.config.cluster_prm_url,
            "cluster_prm_model": self.config.cluster_prm_model,
            "cluster_prm_api_key": self.config.cluster_prm_api_key,
            "evolve_enabled": evolve_enabled,
            "evolve_aggregate_sessions": True,
            "evolve_api_base": self.config.evolve_api_base or self.config.cluster_llm_api_base,
            "evolve_api_key": self.config.evolve_api_key or self.config.cluster_llm_api_key,
            "evolve_model": self.config.evolve_model or self.config.cluster_llm_model_id,
            "evolve_max_tokens": self.config.evolve_max_tokens,
            "evolve_use_success_feedback": self.config.evolve_use_success_feedback,
            "strict_require_nonempty_response": self.config.strict_require_nonempty_response,
            "strict_require_conversation_records": self.config.strict_require_conversation_records,
            "strict_require_skill_reads_from_local_dir": self.config.strict_require_skill_reads_from_local_dir,
            "strict_forbidden_substrings": list(self.config.strict_forbidden_substrings),
        }
        config_path = arm_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return config_path

    def _run_benchmark(self, config_path: Path) -> dict[str, Any]:
        return run_benchmark_from_config(str(config_path))

    def _build_report(
        self,
        *,
        run_root: Path,
        train_task_specs: list[dict[str, str]],
        eval_task_specs: list[dict[str, str]],
        baseline_report: dict[str, Any],
        evolve_report: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_records = _flatten_phase_records(baseline_phase)
        baseline_usage = _aggregate_usage(baseline_records)

        evolve_round = (evolve_report.get("rounds") or [{}])[-1] if (evolve_report.get("rounds") or []) else {}
        train_phase = evolve_round.get("train", {}) or {}
        eval_phase = evolve_round.get("eval", {}) or {}
        train_records = _flatten_phase_records(train_phase)
        eval_records = _flatten_phase_records(eval_phase)
        train_usage = _aggregate_usage(train_records)
        eval_usage = _aggregate_usage(eval_records)
        evolve_info = (evolve_round.get("sync", {}) or {}).get("evolution", {}) or {}
        manifest_path = str(evolve_info.get("manifest_path") or "")
        evolved_skills = _load_manifest_names(manifest_path) if manifest_path else []

        baseline_success = _safe_float((baseline_report.get("summary") or {}).get("initial_eval_success_rate"))
        baseline_score = _safe_float((baseline_report.get("summary") or {}).get("initial_eval_mean_score"))
        evolve_success = _safe_float((evolve_report.get("summary") or {}).get("final_eval_success_rate"))
        evolve_score = _safe_float((evolve_report.get("summary") or {}).get("final_eval_mean_score"))

        return {
            "name": self.config.name,
            "run_root": str(run_root),
            "benchmark_root": str(self.benchmark_root),
            "model": self.config.cluster_llm_model_id,
            "evolution_strategy": {
                "improve_prompt": "conservative_local_edit",
                "use_success_feedback": self.config.evolve_use_success_feedback,
                "legacy_strategy_input": self.config.evolve_strategy,
            },
            "devices": self.config.devices,
            "task_count": len(train_task_specs),
            "train_task_count": len(train_task_specs),
            "eval_task_count": len(eval_task_specs),
            "tasks": train_task_specs,
            "eval_tasks": eval_task_specs,
            "concurrency": {
                "node_parallelism": self.config.devices,
                "max_parallel_devices": self.config.max_parallel_devices or self.config.devices,
                "note": "Current implementation drives one in-flight task per node to keep session artifact attribution correct.",
            },
            "arms": {
                "baseline": {
                    "report_path": str(Path(baseline_report.get("workspace_dir", ".")) / "report.json"),
                    "success_rate": baseline_success,
                    "mean_score": baseline_score,
                    "eval_usage": baseline_usage,
                },
                "evolve_then_broadcast": {
                    "report_path": str(Path(evolve_report.get("workspace_dir", ".")) / "report.json"),
                    "success_rate": evolve_success,
                    "mean_score": evolve_score,
                    "train": {
                        "tasks": train_phase.get("tasks", 0),
                        "success_rate": _safe_float(train_phase.get("success_rate")),
                        "usage": train_usage,
                    },
                    "eval": {
                        "tasks": eval_phase.get("tasks", 0),
                        "success_rate": _safe_float(eval_phase.get("success_rate")),
                        "mean_score": _safe_float(eval_phase.get("mean_score")),
                        "usage": eval_usage,
                    },
                    "evolution": {
                        "skills_evolved": _safe_int((evolve_info.get("summary") or {}).get("skills_evolved")),
                        "failed_turns": _safe_int((evolve_info.get("summary") or {}).get("failed_turns")),
                        "manifest_path": manifest_path,
                        "skills": evolved_skills,
                    },
                },
            },
            "compare": {
                "success_rate_gain": round(evolve_success - baseline_success, 6),
                "mean_score_gain": round(evolve_score - baseline_score, 6),
                "eval_cost_delta_usd": round(eval_usage["cost_usd"] - baseline_usage["cost_usd"], 6),
                "eval_elapsed_delta_s": round(eval_usage["elapsed_time_s"] - baseline_usage["elapsed_time_s"], 3),
                "end_to_end_task_cost_delta_usd": round(
                    (train_usage["cost_usd"] + eval_usage["cost_usd"]) - baseline_usage["cost_usd"],
                    6,
                ),
            },
        }

    def _build_markdown(self, report: dict[str, Any]) -> str:
        baseline = report["arms"]["baseline"]
        evolve = report["arms"]["evolve_then_broadcast"]
        compare = report["compare"]
        lines = [
            f"# {report['name']}",
            "",
            "## Summary",
            "",
            f"- Model: `{report['model']}`",
            f"- Improve prompt: `{report.get('evolution_strategy', {}).get('improve_prompt', 'conservative_local_edit')}`",
            f"- Success feedback enabled: `{bool(report.get('evolution_strategy', {}).get('use_success_feedback', False))}`",
            f"- Devices: {report['devices']}",
            f"- Task count: {report['task_count']}",
            f"- Baseline success: {baseline['success_rate']:.3f}",
            f"- Evolve+broadcast success: {evolve['success_rate']:.3f}",
            f"- Success gain: {compare['success_rate_gain']:+.3f}",
            f"- Baseline mean score: {baseline['mean_score']:.3f}",
            f"- Evolve+broadcast mean score: {evolve['mean_score']:.3f}",
            f"- Mean score gain: {compare['mean_score_gain']:+.3f}",
            f"- Skills evolved: {evolve['evolution']['skills_evolved']}",
            "",
            "## Artifacts",
            "",
            f"- JSON report: `{report['run_root']}/report.json`",
            f"- Baseline report: `{baseline['report_path']}`",
            f"- Evolve+broadcast report: `{evolve['report_path']}`",
        ]
        if evolve["evolution"]["skills"]:
            lines.extend(["", "## Evolved Skills", ""])
            for name in evolve["evolution"]["skills"]:
                lines.append(f"- {name}")
        return "\n".join(lines) + "\n"


def run_wildclawbench_transfer_experiment_from_config(config_path: str) -> dict[str, Any]:
    config = WildClawBenchTransferExperimentConfig.from_file(config_path)
    runner = WildClawBenchTransferExperimentRunner(config)
    return runner.run()
