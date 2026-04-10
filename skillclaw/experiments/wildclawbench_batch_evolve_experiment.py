"""
WildClawBench batch-evolve experiment runner.

This experiment runs a one-shot before/evolve/after comparison on a selected
subset of WildClawBench tasks:

1. before: run the selected tasks in parallel with their default benchmark
   skills
2. evolve: aggregate the resulting sessions and run one evolve pass
3. after: rerun the same task subset in parallel with the broadcast skills
4. compare: report average score gain, per-query deltas, step changes, and the
   evolved skills
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

from .group_benchmark import GroupBenchmarkConfig, GroupBenchmarkRunner, run_benchmark_from_config
from .local_skill_hub import LocalSkillHub
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


def _expand_string_env(value: str) -> str:
    import os

    return os.path.expandvars(value)


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


def _default_evolve_debug_dir(arm_root: Path) -> Path:
    """Store evolve debug dumps alongside the batch run root by default."""
    return (arm_root.parent.parent / "evolve_debug").resolve()


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
        "request_count": 0,
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
        totals["request_count"] += _safe_int(usage.get("request_count"))
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["elapsed_time_s"] = round(totals["elapsed_time_s"], 3)
    return totals


def _parse_task_sections(task_file: Path) -> tuple[dict[str, Any], dict[str, str]]:
    text = task_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    metadata = yaml.safe_load(match.group(1)) if match else {}
    body = match.group(2) if match else text
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = header.group(1).strip()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return metadata or {}, sections


def _strip_codeblock(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        parts = text.splitlines()
        if parts and parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip() == "```":
            parts = parts[:-1]
        text = "\n".join(parts).strip()
    return text


def _task_specific_skill_rank(skills: list[str]) -> int:
    return 0 if any(re.match(r"^\d{2}_task\d+$", name) for name in skills) else 1


def _record_rollout_idx(record: dict[str, Any]) -> int:
    return _safe_int(record.get("rollout_idx", 0))


def _limit_records_by_task_rollouts(records: list[dict[str, Any]], rollout_limit: int) -> list[dict[str, Any]]:
    if rollout_limit <= 0:
        return list(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id") or "")].append(record)

    selected: list[dict[str, Any]] = []
    for task_id in grouped:
        ordered = sorted(
            grouped[task_id],
            key=lambda item: (
                _record_rollout_idx(item),
                str(item.get("device_id") or ""),
                str(item.get("phase") or ""),
            ),
        )
        selected.extend(ordered[:rollout_limit])
    return selected


def _metrics_from_records(records: list[dict[str, Any]]) -> dict[str, float]:
    total_tasks = len(records)
    successes = sum(1 for record in records if bool(record.get("success")))
    scores = [float(record["score"]) for record in records if isinstance(record.get("score"), (int, float))]
    mean_score = _safe_float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "tasks": total_tasks,
        "successes": successes,
        "success_rate": _safe_float(successes / total_tasks) if total_tasks else 0.0,
        "mean_score": mean_score,
    }


@dataclass
class WildClawBenchBatchEvolveExperimentConfig:
    name: str = "wildclawbench-batch-evolve"
    workspace_dir: str = "records/wildclawbench_batch_evolve_experiment"
    resume_run_root: str = ""
    resume_latest: bool = False
    reuse_baseline_from: str = ""
    benchmark_root: str = ""
    selected_task_files: list[str] = field(default_factory=list)
    selection_count: int = 16
    selection_summary_file: str = ""
    selection_max_timeout_seconds: int = 900
    selection_exclude_error_tasks: bool = True
    selection_require_existing_skills: bool = True
    exclude_task_ids: list[str] = field(default_factory=list)
    initial_skills_dir: str = ""
    devices: int = 16
    max_parallel_devices: int = 16
    baseline_rollouts_per_task: int = 1
    after_rollouts_per_task: int = 1
    cluster_configured_nodes: int = 16
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
    cluster_llm_model_id: str = "kimi-k2.5"
    cluster_proxy_api_key: str = "skillclaw"
    cluster_skills_enabled: bool = True
    cluster_retrieval_mode: str = "template"
    cluster_max_context_tokens: int = 20000
    cluster_prm_enabled: bool = False
    cluster_prm_provider: str = "openai"
    cluster_prm_url: str = "https://api.openai.com/v1"
    cluster_prm_model: str = "gpt-5.2"
    cluster_prm_api_key: str = ""
    evolve_api_base: str = ""
    evolve_api_key: str = ""
    evolve_model: str = "gpt-5.4"
    evolve_max_tokens: int = 4096
    evolve_use_success_feedback: bool = True
    cloud_session_payload_mode: str = "minimal"
    wildclawbench_gateway_port: int = 18789
    wildclawbench_container_proxy_host: str = "host.docker.internal"
    wildclawbench_container_proxy_port: str = ""
    wildclawbench_docker_image: str = ""
    wildclawbench_docker_memory_limit: str = ""
    wildclawbench_collect_task_output: bool = False
    wildclawbench_tmp_workspace: str = "/tmp_workspace"
    wildclawbench_task_timeout_seconds_override: int = 0
    wildclawbench_success_threshold: float = 0.5
    wildclawbench_use_score_as_prm: bool = True
    strict_require_nonempty_response: bool = False
    strict_require_conversation_records: bool = False
    strict_require_skill_reads_from_local_dir: bool = False
    strict_forbidden_substrings: list[str] = field(default_factory=list)
    # Iterative evolve settings (used by wildclawbench_iterative_evolve.py)
    iterative_rounds: int = 3
    iterative_resume_round: int = 0
    iterative_import_from: str = ""
    rollouts_per_task: int = 1

    @classmethod
    def from_file(cls, path: str) -> "WildClawBenchBatchEvolveExperimentConfig":
        cfg_path = Path(path).expanduser().resolve()

        # Auto-load .env files so that ${VAR} references in YAML configs
        # resolve correctly even when the caller didn't `export` them.
        # Search order: workspace root (.env next to the config), then
        # common project-level .env locations.
        try:
            from dotenv import load_dotenv

            for candidate in (
                cfg_path.parent / ".env",          # next to the YAML
                cfg_path.parent.parent / ".env",    # one level up
                Path("/root/SkillClawEnv/.env"),    # project root
            ):
                if candidate.is_file():
                    load_dotenv(candidate, override=False)
                    break
        except ImportError:
            pass

        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("wildclawbench batch evolve config must deserialize to a dict")

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
            "resume_run_root",
            "reuse_baseline_from",
            "benchmark_root",
            "selection_summary_file",
            "initial_skills_dir",
            "cluster_seed_openclaw_dir",
            "cluster_openclaw_bin",
            "cluster_skillclaw_bin",
        ):
            value = data.get(key)
            if isinstance(value, str) and value:
                path_obj = Path(value).expanduser()
                if path_obj.is_absolute():
                    data[key] = str(path_obj.resolve())
                else:
                    data[key] = str((base_dir / path_obj).resolve())

        benchmark_root_value = str(data.get("benchmark_root") or "").strip()
        benchmark_root = Path(benchmark_root_value).expanduser() if benchmark_root_value else None

        resolved_tasks: list[str] = []
        for raw_task in data.get("selected_task_files", []) or []:
            value = _expand_string_env(str(raw_task))
            path_obj = Path(value).expanduser()
            if path_obj.is_absolute():
                resolved_tasks.append(str(path_obj.resolve()))
            elif benchmark_root is not None:
                resolved_tasks.append(str((benchmark_root / path_obj).resolve()))
            else:
                resolved_tasks.append(str((base_dir / path_obj).resolve()))
        data["selected_task_files"] = resolved_tasks

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


class WildClawBenchBatchEvolveExperimentRunner:
    def __init__(self, config: WildClawBenchBatchEvolveExperimentConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir).expanduser().resolve()
        self.benchmark_root = Path(config.benchmark_root).expanduser().resolve()
        self.skills_root = self.benchmark_root / "skills"

    def run(self) -> dict[str, Any]:
        run_root = self._resolve_run_root()
        existing_top_report = self._load_json(run_root / "report.json")
        if existing_top_report:
            return existing_top_report

        if self.config.reuse_baseline_from:
            self._copy_baseline_artifacts(run_root)

        selected_task_specs = self._load_or_select_task_specs(run_root)
        if not selected_task_specs:
            raise ValueError("no task selected for batch evolve experiment")

        initial_skills_dir = self._ensure_initial_skills_dir(run_root, selected_task_specs)
        baseline_report = self._load_json(self._baseline_report_path(run_root))
        if not baseline_report:
            baseline_report = self._run_baseline_arm(run_root, selected_task_specs, initial_skills_dir)

        train_summary = self._build_reused_train_summary(baseline_report)
        evolution_summary = self._load_reused_evolution_summary(run_root, train_summary=train_summary)
        if not evolution_summary:
            evolution_summary = self._run_reused_evolution(
                arm_root=run_root / "arms" / "batch_evolve",
                arm_devices=self._arm_device_count(
                    selected_task_specs,
                    self.config.baseline_rollouts_per_task,
                ),
                train_summary=train_summary,
                initial_skills_dir=initial_skills_dir,
            )

        after_skills_dir, pull_summary, remote_skill_count = self._materialize_after_skills(
            arm_root=run_root / "arms" / "batch_evolve",
            initial_skills_dir=initial_skills_dir,
        )
        after_report = self._load_json(self._after_report_path(run_root))
        if not after_report:
            after_report = self._run_after_arm(run_root, selected_task_specs, after_skills_dir)

        evolve_report = self._compose_reused_evolve_report(
            arm_root=run_root / "arms" / "batch_evolve",
            train_summary=train_summary,
            evolution_summary=evolution_summary,
            after_report=after_report,
            pull_summary=pull_summary,
            remote_skill_count=remote_skill_count,
        )

        report = self._build_report(
            run_root=run_root,
            selected_task_specs=selected_task_specs,
            baseline_report=baseline_report,
            evolve_report=evolve_report,
        )
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "report.md").write_text(self._build_markdown(report), encoding="utf-8")
        return report

    def _resolve_run_root(self) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.config.resume_run_root:
            run_root = Path(self.config.resume_run_root).expanduser().resolve()
            if not run_root.exists():
                raise FileNotFoundError(f"resume_run_root not found: {run_root}")
            return run_root
        if self.config.resume_latest:
            latest_incomplete = self._find_latest_incomplete_run()
            if latest_incomplete is not None:
                return latest_incomplete
        run_root = self.workspace_dir / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root

    def _find_latest_incomplete_run(self) -> Path | None:
        if not self.workspace_dir.exists():
            return None
        candidates = [path for path in self.workspace_dir.iterdir() if path.is_dir()]
        for path in sorted(candidates, reverse=True):
            if not (path / "report.json").exists():
                return path
        return None

    def _load_or_select_task_specs(self, run_root: Path) -> list[dict[str, Any]]:
        selected_path = run_root / "selected_tasks.json"
        existing = self._load_json(selected_path)
        if isinstance(existing, list):
            return existing
        selected_task_specs = self._select_task_specs()
        selected_path.write_text(
            json.dumps(selected_task_specs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return selected_task_specs

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, (dict, list)):
            return payload
        return None

    @staticmethod
    def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    @staticmethod
    def _baseline_report_path(run_root: Path) -> Path:
        return run_root / "arms" / "baseline" / "workspace" / "report.json"

    @staticmethod
    def _after_report_path(run_root: Path) -> Path:
        return run_root / "arms" / "batch_evolve" / "workspace" / "report.json"

    def _copy_baseline_artifacts(self, run_root: Path) -> None:
        """Copy selected_tasks.json, initial_skills/, and arms/baseline/ from a
        previously completed run so that only evolve + after need to execute."""
        reuse_src = Path(self.config.reuse_baseline_from).expanduser().resolve()
        if not reuse_src.exists():
            raise FileNotFoundError(f"reuse_baseline_from not found: {reuse_src}")
        for name in ("selected_tasks.json",):
            src_file = reuse_src / name
            dst_file = run_root / name
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        for name in ("initial_skills", "arms/baseline"):
            src_dir = reuse_src / name
            dst_dir = run_root / name
            if src_dir.exists() and not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)

    def _load_reused_evolution_summary(
        self,
        run_root: Path,
        *,
        train_summary: dict[str, Any],
    ) -> dict[str, Any]:
        arm_root = run_root / "arms" / "batch_evolve"
        workspace_root = arm_root / "workspace"
        history_path = workspace_root / "evolve" / "round-1" / "evolve_history.jsonl"
        processed_path = workspace_root / "evolve" / "round-1" / "evolve_processed.json"
        backend_root = workspace_root / "evolve" / "round-1" / "backend"
        group_id = f"{self.config.name}-batch_evolve"
        cloud_group = arm_root / "cloud" / group_id
        manifest_path = cloud_group / "manifest.jsonl"
        registry_path = cloud_group / "evolve_skill_registry.json"
        if not history_path.exists() and not processed_path.exists() and not registry_path.exists() and not manifest_path.exists():
            return {}

        history_records = self._load_jsonl_records(history_path)
        summary = history_records[-1] if history_records else {}
        aggregated_session_paths = sorted(str(path) for path in (backend_root / group_id / "sessions").glob("*.json"))
        return {
            "round": 1,
            "summary": summary,
            "manifest_path": str(manifest_path),
            "registry_path": str(registry_path),
            "aggregation": {
                "enabled": True,
                "backend_root": str(backend_root),
                "source_session_count": sum(len(device.get("records", []) or []) for device in train_summary.get("devices", [])),
                "aggregated_session_count": len(aggregated_session_paths),
                "aggregated_session_paths": aggregated_session_paths,
            },
        }

    def _load_summary_by_task(self) -> dict[str, dict[str, Any]]:
        path = str(self.config.selection_summary_file or "").strip()
        if not path:
            return {}
        summary_path = Path(path)
        if not summary_path.exists():
            return {}
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        by_task: dict[str, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or item.get("id") or "").strip()
            if not task_id:
                continue
            base = re.sub(r"_\d{8}_\d+$", "", task_id)
            if base == task_id:
                base = re.sub(r"_\d{8}_\d{4}$", "", task_id)
            by_task[base] = item
        return by_task

    def _select_task_specs(self) -> list[dict[str, Any]]:
        summary_by_task = self._load_summary_by_task()
        if self.config.selected_task_files:
            task_files = [Path(item).expanduser().resolve() for item in self.config.selected_task_files]
            return [self._build_task_spec(task_file, summary_by_task=summary_by_task) for task_file in task_files]

        excluded = {str(task_id).strip() for task_id in self.config.exclude_task_ids if str(task_id).strip()}
        all_specs: list[dict[str, Any]] = []
        for task_file in sorted((self.benchmark_root / "tasks").glob("*/*.md")):
            spec = self._build_task_spec(task_file, summary_by_task=summary_by_task)
            if spec["task_id"] in excluded:
                continue
            all_specs.append(spec)

        count = max(1, int(self.config.selection_count or 16))
        if count >= len(all_specs):
            return all_specs

        candidates: list[dict[str, Any]] = []
        for spec in all_specs:
            if self.config.selection_require_existing_skills and not spec["required_skills"]:
                continue
            if self.config.selection_max_timeout_seconds > 0 and spec["timeout_seconds"] > self.config.selection_max_timeout_seconds:
                continue
            if self.config.selection_exclude_error_tasks and spec["historical"]["error"]:
                continue
            candidates.append(spec)

        candidates.sort(
            key=lambda item: (
                _task_specific_skill_rank(item["required_skills"]),
                item["timeout_seconds"],
                item["historical"]["score"] is None,
                item["historical"]["score"] if item["historical"]["score"] is not None else 999.0,
                item["historical"]["elapsed_time_s"] if item["historical"]["elapsed_time_s"] is not None else 9999.0,
                item["historical"]["request_count"] if item["historical"]["request_count"] is not None else 9999,
                item["task_id"],
            )
        )
        return candidates[:count]

    def _build_task_spec(
        self,
        task_file: Path,
        *,
        summary_by_task: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        parsed = parse_wildclaw_task_md(task_file, benchmark_root=self.benchmark_root)
        _, sections = _parse_task_sections(task_file)
        skills = [
            line.strip()
            for line in _strip_codeblock(sections.get("Skills", "")).splitlines()
            if line.strip()
        ]
        required_skills = [skill for skill in skills if (self.skills_root / skill).exists()]
        historical = summary_by_task.get(parsed.task_id, {})
        usage = historical.get("usage") or {}
        score = (historical.get("scores") or {}).get("overall_score")
        return {
            "task_id": parsed.task_id,
            "task_file": parsed.file_path,
            "category": parsed.category,
            "timeout_seconds": parsed.timeout_seconds,
            "required_skills": required_skills,
            "historical": {
                "score": _safe_float(score) if isinstance(score, (int, float)) else (float(score) if isinstance(score, str) and score.strip() else None),
                "elapsed_time_s": _safe_float(usage.get("elapsed_time")) if usage else None,
                "request_count": _safe_int(usage.get("request_count")) if usage else None,
                "error": str(historical.get("error") or "").strip(),
            },
        }

    def _ensure_initial_skills_dir(self, run_root: Path, task_specs: list[dict[str, Any]]) -> Path:
        target_dir = run_root / "initial_skills"
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.config.initial_skills_dir:
            src = Path(self.config.initial_skills_dir).expanduser().resolve()
            if src.exists():
                shutil.copytree(src, target_dir, dirs_exist_ok=True)
        copied: set[str] = set()
        for spec in task_specs:
            for skill_name in spec.get("required_skills", []) or []:
                if skill_name in copied:
                    continue
                src_dir = self.skills_root / skill_name
                if src_dir.exists():
                    shutil.copytree(src_dir, target_dir / skill_name, dirs_exist_ok=True)
                    copied.add(skill_name)
        return target_dir

    def _arm_device_count(self, task_specs: list[dict[str, Any]], rollouts_per_task: int) -> int:
        total_task_runs = max(1, len(task_specs) * max(1, int(rollouts_per_task or 1)))
        configured = int(self.config.devices or 0)
        if configured > 0:
            return min(configured, total_task_runs)
        return total_task_runs

    def _run_baseline_arm(
        self,
        run_root: Path,
        task_specs: list[dict[str, Any]],
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
            arm_devices=self._arm_device_count(task_specs, self.config.baseline_rollouts_per_task),
            rollouts_per_task=max(1, self.config.baseline_rollouts_per_task),
            rounds=0,
            sync_enabled=False,
            evolve_enabled=False,
            eval_every_round=False,
            initial_eval_enabled=True,
            device_field="",
            port_offset=0,
        )
        return self._run_benchmark(config_path)

    def _run_evolve_arm(
        self,
        run_root: Path,
        task_specs: list[dict[str, Any]],
        initial_skills_dir: Path,
        baseline_report: dict[str, Any],
    ) -> dict[str, Any]:
        arm_root = run_root / "arms" / "batch_evolve"
        train_summary = self._build_reused_train_summary(baseline_report)
        evolution_summary = self._run_reused_evolution(
            arm_root=arm_root,
            arm_devices=self._arm_device_count(task_specs, self.config.baseline_rollouts_per_task),
            train_summary=train_summary,
            initial_skills_dir=initial_skills_dir,
        )
        after_skills_dir, pull_summary, remote_skill_count = self._materialize_after_skills(
            arm_root=arm_root,
            initial_skills_dir=initial_skills_dir,
        )
        after_report = self._run_after_arm(run_root, task_specs, after_skills_dir)
        return self._compose_reused_evolve_report(
            arm_root=arm_root,
            train_summary=train_summary,
            evolution_summary=evolution_summary,
            after_report=after_report,
            pull_summary=pull_summary,
            remote_skill_count=remote_skill_count,
        )

    def _build_reused_train_summary(self, baseline_report: dict[str, Any]) -> dict[str, Any]:
        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_records = _limit_records_by_task_rollouts(
            _flatten_phase_records(baseline_phase),
            max(1, self.config.baseline_rollouts_per_task),
        )
        grouped_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in baseline_records:
            grouped_by_device[str(record.get("device_id") or "")].append(record)

        devices_summary: list[dict[str, Any]] = []
        for device_id, raw_records in grouped_by_device.items():
            train_records: list[dict[str, Any]] = []
            scores: list[float] = []
            successes = 0
            for record in raw_records:
                cloned = dict(record)
                cloned["phase"] = "train"
                cloned["round"] = 1
                train_records.append(cloned)
                if bool(record.get("success")):
                    successes += 1
                if isinstance(record.get("score"), (int, float)):
                    scores.append(float(record["score"]))
            devices_summary.append(
                {
                    "device_id": device_id,
                    "tasks": len(train_records),
                    "successes": successes,
                    "success_rate": _safe_float(successes / len(train_records)) if train_records else 0.0,
                    "new_skills": 0,
                    "local_skill_count": {},
                    "total_score": sum(scores),
                    "scored_tasks": len(scores),
                    "mean_score": _safe_float(sum(scores) / len(scores)) if scores else 0.0,
                    "records": train_records,
                }
            )

        total_tasks = sum(int(item.get("tasks", 0)) for item in devices_summary)
        total_successes = sum(int(item.get("successes", 0)) for item in devices_summary)
        total_score = sum(float(item.get("total_score", 0.0)) for item in devices_summary)
        scored_tasks = sum(int(item.get("scored_tasks", 0)) for item in devices_summary)
        return {
            "devices": devices_summary,
            "tasks": total_tasks,
            "successes": total_successes,
            "success_rate": _safe_float(total_successes / total_tasks) if total_tasks else 0.0,
            "new_skills": 0,
            "total_score": total_score,
            "scored_tasks": scored_tasks,
            "mean_score": _safe_float(total_score / scored_tasks) if scored_tasks else 0.0,
        }

    def _run_reused_evolution(
        self,
        *,
        arm_root: Path,
        arm_devices: int,
        train_summary: dict[str, Any],
        initial_skills_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        arm_root.mkdir(parents=True, exist_ok=True)

        # ---- seed initial skills into cloud storage ------------------- #
        # In a normal run the baseline devices push their skills to cloud
        # after each task.  In the "reused evolution" path we skip the
        # baseline execution, so the cloud has only sessions, no skills.
        # Push the initial skills here so the evolve server can fetch
        # current_skill content for each skill referenced by sessions.
        group_id = f"{self.config.name}-batch_evolve"
        if initial_skills_dir and initial_skills_dir.exists():
            hub = LocalSkillHub(
                root_dir=str(arm_root / "cloud"),
                group_id=group_id,
                user_alias="seed",
            )
            push_result = hub.push_skills(str(initial_skills_dir))
            logger.info(
                "[BatchEvolve] seeded initial skills into cloud: %s", push_result,
            )
        # --------------------------------------------------------------- #

        runner = GroupBenchmarkRunner(
            GroupBenchmarkConfig(
                name=f"{self.config.name}-batch_evolve",
                workspace_dir=str(arm_root / "workspace"),
                tasks_path=str(arm_root / "tasks.jsonl"),
                cloud_dir=str(arm_root / "cloud"),
                group_id=f"{self.config.name}-batch_evolve",
                devices=arm_devices,
                cluster_configured_nodes=max(self.config.cluster_configured_nodes, arm_devices),
                cluster_llm_api_base=self.config.cluster_llm_api_base,
                cluster_llm_api_key=self.config.cluster_llm_api_key,
                cluster_llm_model_id=self.config.cluster_llm_model_id,
                evolve_api_base=self.config.evolve_api_base or self.config.cluster_llm_api_base,
                evolve_api_key=self.config.evolve_api_key or self.config.cluster_llm_api_key,
                evolve_model=self.config.evolve_model or self.config.cluster_llm_model_id,
                evolve_max_tokens=self.config.evolve_max_tokens,
                evolve_debug_dir=str(_default_evolve_debug_dir(arm_root)),
                evolve_use_success_feedback=self.config.evolve_use_success_feedback,
                evolve_aggregate_sessions=True,
                executor="replay",
            )
        )
        debug_dir = _default_evolve_debug_dir(arm_root)
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[BatchEvolve] evolve debug dumps -> %s",
            debug_dir,
        )
        return runner._run_evolution(1, train_summary=train_summary)

    def _materialize_after_skills(
        self,
        *,
        arm_root: Path,
        initial_skills_dir: Path,
    ) -> tuple[Path, dict[str, int], int]:
        after_skills_dir = arm_root / "after_skills"
        if after_skills_dir.exists():
            shutil.rmtree(after_skills_dir)
        after_skills_dir.mkdir(parents=True, exist_ok=True)
        if initial_skills_dir.exists():
            shutil.copytree(initial_skills_dir, after_skills_dir, dirs_exist_ok=True)

        hub = LocalSkillHub(
            root_dir=str(arm_root / "cloud"),
            group_id=f"{self.config.name}-batch_evolve",
            user_alias="benchmark",
        )
        pull_summary = hub.pull_skills(str(after_skills_dir))
        return after_skills_dir, pull_summary, len(hub.list_remote())

    def _run_after_arm(
        self,
        run_root: Path,
        task_specs: list[dict[str, Any]],
        initial_skills_dir: Path,
    ) -> dict[str, Any]:
        arm_root = run_root / "arms" / "batch_evolve"
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
            arm_name="batch_evolve",
            tasks_path=tasks_path,
            initial_skills_dir=initial_skills_dir,
            arm_devices=self._arm_device_count(task_specs, self.config.after_rollouts_per_task),
            rollouts_per_task=max(1, self.config.after_rollouts_per_task),
            rounds=0,
            sync_enabled=False,
            evolve_enabled=False,
            eval_every_round=False,
            initial_eval_enabled=True,
            device_field="",
            port_offset=200,
        )
        return self._run_benchmark(config_path)

    def _compose_reused_evolve_report(
        self,
        *,
        arm_root: Path,
        train_summary: dict[str, Any],
        evolution_summary: dict[str, Any],
        after_report: dict[str, Any],
        pull_summary: dict[str, int],
        remote_skill_count: int,
    ) -> dict[str, Any]:
        after_phase = after_report.get("initial_eval", {}) or {}
        after_records = _limit_records_by_task_rollouts(
            _flatten_phase_records(after_phase),
            max(1, self.config.after_rollouts_per_task),
        )
        after_metrics = _metrics_from_records(after_records)
        return {
            "workspace_dir": str(after_report.get("workspace_dir") or (arm_root / "workspace")),
            "summary": {
                "initial_eval_success_rate": 0.0,
                "final_eval_success_rate": after_metrics["success_rate"],
                "initial_eval_mean_score": 0.0,
                "final_eval_mean_score": after_metrics["mean_score"],
            },
            "rounds": [
                {
                    "round": 1,
                    "train": train_summary,
                    "sync": {
                        "push": [],
                        "evolution": evolution_summary,
                        "pull": [{"device_id": "reused-before-eval", **pull_summary}],
                        "remote_skill_count": remote_skill_count,
                    },
                    "eval": after_phase,
                    "feedback": after_report.get("feedback", {}),
                }
            ],
        }

    def _write_arm_config(
        self,
        *,
        arm_root: Path,
        arm_name: str,
        tasks_path: Path,
        initial_skills_dir: Path,
        arm_devices: int,
        rollouts_per_task: int,
        rounds: int,
        sync_enabled: bool,
        evolve_enabled: bool,
        eval_every_round: bool,
        initial_eval_enabled: bool,
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
            "group_id": f"{self.config.name}-{arm_name}",
            "initial_skills_dir": str(initial_skills_dir),
            "devices": arm_devices,
            "eval_devices": arm_devices,
            "cluster_configured_nodes": max(self.config.cluster_configured_nodes, arm_devices),
            "rounds": rounds,
            "initial_eval_enabled": initial_eval_enabled,
            "train_tasks_per_device_per_round": 1,
            "max_parallel_devices": max_parallel,
            "sync_enabled": sync_enabled,
            "eval_every_round": eval_every_round,
            "eval_assignment_strategy": "distribute",
            "device_field": device_field,
            "executor": "wildclawbench_cluster",
            "wildclawbench_root": str(self.benchmark_root),
            "wildclawbench_output_dir": str(arm_root / "wildclawbench_output"),
            "wildclawbench_container_proxy_host": self.config.wildclawbench_container_proxy_host,
            "wildclawbench_container_proxy_port": self.config.wildclawbench_container_proxy_port,
            "wildclawbench_gateway_port": self.config.wildclawbench_gateway_port,
            "wildclawbench_docker_image": self.config.wildclawbench_docker_image,
            "wildclawbench_docker_memory_limit": self.config.wildclawbench_docker_memory_limit,
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
            "cluster_skills_enabled": self.config.cluster_skills_enabled,
            "cluster_retrieval_mode": self.config.cluster_retrieval_mode,
            "cluster_max_context_tokens": self.config.cluster_max_context_tokens,
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
            "cloud_session_payload_mode": self.config.cloud_session_payload_mode,
            "strict_require_nonempty_response": self.config.strict_require_nonempty_response,
            "strict_require_conversation_records": self.config.strict_require_conversation_records,
            "strict_require_skill_reads_from_local_dir": self.config.strict_require_skill_reads_from_local_dir,
            "strict_forbidden_substrings": list(self.config.strict_forbidden_substrings),
            "wildclawbench_collect_task_output": self.config.wildclawbench_collect_task_output,
            "resume_completed_tasks": True,
            "rollouts_per_task": max(1, int(rollouts_per_task or 1)),
        }
        config_path = arm_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return config_path

    def _run_benchmark(self, config_path: Path) -> dict[str, Any]:
        return run_benchmark_from_config(str(config_path))

    def _record_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        wildbench = ((record.get("artifacts") or {}).get("wildclawbench") or {})
        usage = wildbench.get("usage") or {}
        return {
            "task_id": str(record.get("task_id") or ""),
            "rollout_idx": _record_rollout_idx(record),
            "success": bool(record.get("success")),
            "score": _safe_float(record.get("score")),
            "used_skills": [str(item) for item in record.get("used_skills", []) or [] if str(item).strip()],
            "latency_ms": _safe_float(record.get("latency_ms")),
            "step_count": _safe_int(usage.get("request_count")),
            "elapsed_time_s": _safe_float(usage.get("elapsed_time")),
            "score_breakdown": dict(record.get("score_breakdown") or {}),
        }

    @staticmethod
    def _aggregate_rollout_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
        if not summaries:
            return {}
        if len(summaries) == 1:
            return summaries[0]

        scores = [summary["score"] for summary in summaries if isinstance(summary.get("score"), (int, float))]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        score_std = 0.0
        if len(scores) > 1:
            score_std = (sum((score - mean_score) ** 2 for score in scores) / len(scores)) ** 0.5
        success_count = sum(1 for summary in summaries if summary.get("success"))
        step_counts = [summary.get("step_count", 0) for summary in summaries]
        elapsed_times = [summary.get("elapsed_time_s", 0.0) for summary in summaries]

        best = max(summaries, key=lambda summary: (summary.get("score", 0.0), 1 if summary.get("success") else 0))
        result = dict(best)
        result["score"] = round(mean_score, 6)
        result["success"] = success_count > len(summaries) / 2
        result["rollout_count"] = len(summaries)
        result["rollout_scores"] = scores
        result["score_std"] = round(score_std, 4)
        result["success_count"] = success_count
        result["step_count"] = round(sum(step_counts) / len(step_counts)) if step_counts else 0
        result["elapsed_time_s"] = round(sum(elapsed_times) / len(elapsed_times), 3) if elapsed_times else 0.0
        return result

    def _group_records_by_task(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("task_id") or "")].append(self._record_summary(record))
        return {
            task_id: self._aggregate_rollout_summaries(summaries)
            for task_id, summaries in grouped.items()
        }

    def _build_report(
        self,
        *,
        run_root: Path,
        selected_task_specs: list[dict[str, Any]],
        baseline_report: dict[str, Any],
        evolve_report: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_records = _limit_records_by_task_rollouts(
            _flatten_phase_records(baseline_phase),
            max(1, self.config.baseline_rollouts_per_task),
        )
        baseline_usage = _aggregate_usage(baseline_records)
        baseline_metrics = _metrics_from_records(baseline_records)

        evolve_round = (evolve_report.get("rounds") or [{}])[-1] if (evolve_report.get("rounds") or []) else {}
        train_phase = evolve_round.get("train", {}) or {}
        eval_phase = evolve_round.get("eval", {}) or {}
        train_records = _flatten_phase_records(train_phase)
        after_records = _limit_records_by_task_rollouts(
            _flatten_phase_records(eval_phase),
            max(1, self.config.after_rollouts_per_task),
        )
        train_usage = _aggregate_usage(train_records)
        after_usage = _aggregate_usage(after_records)
        after_metrics = _metrics_from_records(after_records)
        evolve_info = (evolve_round.get("sync", {}) or {}).get("evolution", {}) or {}
        manifest_path = str(evolve_info.get("manifest_path") or "")
        evolved_skills = _load_manifest_names(manifest_path) if manifest_path else []

        before_by_task = self._group_records_by_task(baseline_records)
        after_by_task = self._group_records_by_task(after_records)

        per_query: list[dict[str, Any]] = []
        for spec in selected_task_specs:
            task_id = spec["task_id"]
            before = before_by_task.get(task_id, {})
            after = after_by_task.get(task_id, {})
            before_score = _safe_float(before.get("score"))
            after_score = _safe_float(after.get("score"))
            before_steps = _safe_int(before.get("step_count"))
            after_steps = _safe_int(after.get("step_count"))
            per_query.append(
                {
                    "task_id": task_id,
                    "category": spec["category"],
                    "required_skills": list(spec.get("required_skills", []) or []),
                    "before": before,
                    "after": after,
                    "score_gain": round(after_score - before_score, 6),
                    "step_delta": after_steps - before_steps,
                    "improved": after_score > before_score,
                    "success_flipped": (not bool(before.get("success"))) and bool(after.get("success")),
                }
            )

        baseline_success = baseline_metrics["success_rate"]
        baseline_score = baseline_metrics["mean_score"]
        after_success = after_metrics["success_rate"]
        after_score = after_metrics["mean_score"]

        evolve_workspace = Path(str(evolve_report.get("workspace_dir") or "."))
        history_path = evolve_workspace / "evolve" / "round-1" / "evolve_history.jsonl"
        processed_path = evolve_workspace / "evolve" / "round-1" / "evolve_processed.json"

        return {
            "name": self.config.name,
            "run_root": str(run_root),
            "benchmark_root": str(self.benchmark_root),
            "agent_model": self.config.cluster_llm_model_id,
            "evolve_model": self.config.evolve_model,
            "selection": {
                "selection_count": len(selected_task_specs),
                "selection_summary_file": self.config.selection_summary_file,
                "max_timeout_seconds": self.config.selection_max_timeout_seconds,
                "tasks": selected_task_specs,
            },
            "concurrency": {
                "devices": max(
                    self._arm_device_count(selected_task_specs, self.config.baseline_rollouts_per_task),
                    self._arm_device_count(selected_task_specs, self.config.after_rollouts_per_task),
                ),
                "baseline_devices": self._arm_device_count(selected_task_specs, self.config.baseline_rollouts_per_task),
                "after_devices": self._arm_device_count(selected_task_specs, self.config.after_rollouts_per_task),
                "max_parallel_devices": self.config.max_parallel_devices or self._arm_device_count(
                    selected_task_specs,
                    self.config.baseline_rollouts_per_task,
                ),
                "assignment": "one task per device per phase",
                "baseline_rollouts_per_task": max(1, self.config.baseline_rollouts_per_task),
                "after_rollouts_per_task": max(1, self.config.after_rollouts_per_task),
            },
            "arms": {
                "before": {
                    "report_path": str(Path(baseline_report.get("workspace_dir", ".")) / "report.json"),
                    "success_rate": baseline_success,
                    "mean_score": baseline_score,
                    "usage": baseline_usage,
                },
                "batch_evolve": {
                    "report_path": str(Path(evolve_report.get("workspace_dir", ".")) / "report.json"),
                    "success_rate": after_success,
                    "mean_score": after_score,
                    "train_usage": train_usage,
                    "after_usage": after_usage,
                },
            },
            "compare": {
                "success_rate_gain": round(after_success - baseline_success, 6),
                "mean_score_gain": round(after_score - baseline_score, 6),
                "eval_cost_delta_usd": round(after_usage["cost_usd"] - baseline_usage["cost_usd"], 6),
                "eval_elapsed_delta_s": round(after_usage["elapsed_time_s"] - baseline_usage["elapsed_time_s"], 3),
            },
            "per_query": per_query,
            "skill_evolution": {
                "summary": evolve_info.get("summary") or {},
                "manifest_path": manifest_path,
                "history_path": str(history_path),
                "processed_path": str(processed_path),
                "skills": evolved_skills,
            },
        }

    def _build_markdown(self, report: dict[str, Any]) -> str:
        before = report["arms"]["before"]
        after = report["arms"]["batch_evolve"]
        compare = report["compare"]
        lines = [
            f"# {report['name']}",
            "",
            "## Summary",
            "",
            f"- Agent model: `{report['agent_model']}`",
            f"- Evolve model: `{report['evolve_model']}`",
            f"- Task count: {report['selection']['selection_count']}",
            f"- Concurrency: {report['concurrency']['devices']}",
            f"- Before mean score: {before['mean_score']:.3f}",
            f"- After mean score: {after['mean_score']:.3f}",
            f"- Mean score gain: {compare['mean_score_gain']:+.3f}",
            f"- Before success rate: {before['success_rate']:.3f}",
            f"- After success rate: {after['success_rate']:.3f}",
            f"- Success rate gain: {compare['success_rate_gain']:+.3f}",
            f"- Evolved skills: {', '.join(report['skill_evolution']['skills']) or '(none)'}",
            "",
            "## Per Query",
            "",
            "| Task | Before | After | Gain | Step Δ |",
            "|------|--------|-------|------|--------|",
        ]
        for item in report["per_query"]:
            lines.append(
                "| {task} | {before:.3f} | {after:.3f} | {gain:+.3f} | {step_delta:+d} |".format(
                    task=item["task_id"],
                    before=_safe_float((item.get("before") or {}).get("score")),
                    after=_safe_float((item.get("after") or {}).get("score")),
                    gain=_safe_float(item.get("score_gain")),
                    step_delta=_safe_int(item.get("step_delta")),
                )
            )
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                f"- JSON report: `{report['run_root']}/report.json`",
                f"- Before report: `{before['report_path']}`",
                f"- After report: `{after['report_path']}`",
                f"- Evolve history: `{report['skill_evolution']['history_path']}`",
            ]
        )
        return "\n".join(lines) + "\n"


def run_wildclawbench_batch_evolve_experiment_from_config(config_path: str) -> dict[str, Any]:
    config = WildClawBenchBatchEvolveExperimentConfig.from_file(config_path)
    runner = WildClawBenchBatchEvolveExperimentRunner(config)
    return runner.run()
