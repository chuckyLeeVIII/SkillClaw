"""
WildClawBench iterative multi-round evolve experiment runner.

Extends the one-shot batch-evolve experiment into a loop:

    baseline-eval -> evolve -> eval -> evolve -> eval -> ... (N rounds)

Each round:
  1. Takes the previous round's eval sessions as training data.
  2. Runs one evolution pass (skill creation/editing).
  3. Materializes evolved skills on top of the previous round's skill set.
  4. Re-evaluates all tasks with the updated skills.

Supports checkpoint/resume: each completed round is recorded in
``iterative_progress.json`` so the experiment can be stopped and restarted
from the last completed round.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .group_benchmark import GroupBenchmarkConfig, GroupBenchmarkRunner
from .local_skill_hub import LocalSkillHub
from .wildclawbench_batch_evolve_experiment import (
    WildClawBenchBatchEvolveExperimentConfig,
    WildClawBenchBatchEvolveExperimentRunner,
    _append_jsonl,
    _flatten_phase_records,
    _limit_records_by_task_rollouts,
    _load_manifest_names,
    _safe_float,
)


class IterativeEvolveRunner:
    """Multi-round eval-evolve loop on WildClawBench tasks."""

    def __init__(self, config: WildClawBenchBatchEvolveExperimentConfig):
        self.config = config
        # When a unified rollouts_per_task is set (e.g. from iterative YAML),
        # propagate it to the baseline / after rollout fields used by the inner
        # batch-evolve runner, unless they were explicitly overridden.
        rpt = max(1, int(config.rollouts_per_task or 1))
        if rpt > 1:
            if config.baseline_rollouts_per_task <= 1:
                config.baseline_rollouts_per_task = rpt
            if config.after_rollouts_per_task <= 1:
                config.after_rollouts_per_task = rpt
        self.inner = WildClawBenchBatchEvolveExperimentRunner(config)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full iterative experiment and return the aggregated report."""

        # If importing from a previous batch_evolve run, set up the new run_root first.
        if self.config.iterative_import_from:
            run_root = self._import_from_batch_evolve(
                Path(self.config.iterative_import_from).expanduser().resolve()
            )
        else:
            run_root = self.inner._resolve_run_root()

        # If a final report already exists, return it directly.
        existing = self._load_json(run_root / "report.json")
        if existing and isinstance(existing, dict):
            completed = existing.get("rounds_completed", 0)
            if completed >= self.config.iterative_rounds:
                return existing

        # Prepare tasks and initial skills (with caching).
        task_specs = self.inner._load_or_select_task_specs(run_root)
        if not task_specs:
            raise ValueError("no tasks selected for iterative evolve experiment")
        initial_skills_dir = self.inner._ensure_initial_skills_dir(run_root, task_specs)

        # Load checkpoint progress.
        progress = self._load_progress(run_root)

        # ---- Round 0: baseline evaluation ----
        baseline_report_path = run_root / "arms" / "baseline" / "workspace" / "report.json"
        if progress.get("baseline_done"):
            baseline_report = self._load_json(baseline_report_path)
            if baseline_report is None:
                raise RuntimeError(
                    f"progress says baseline is done but report is missing: {baseline_report_path}"
                )
        else:
            baseline_report = self.inner._run_baseline_arm(run_root, task_specs, initial_skills_dir)
            progress["baseline_done"] = True
            progress["baseline_report_path"] = str(baseline_report_path)
            self._save_progress(run_root, progress)

        # ---- Iterative rounds ----
        prev_report = baseline_report
        prev_skills_dir = initial_skills_dir

        completed_rounds: list[int] = progress.get("completed_rounds", [])
        total_rounds = self.config.iterative_rounds

        for round_idx in range(1, total_rounds + 1):
            round_dir = run_root / "arms" / f"round-{round_idx}"

            if round_idx in completed_rounds:
                # Checkpoint: load prior round's state and skip.
                saved_eval_report = self._load_json(
                    round_dir / "eval" / "workspace" / "report.json"
                )
                if saved_eval_report is None:
                    # Checkpoint is inconsistent; re-run this round.
                    completed_rounds = [r for r in completed_rounds if r != round_idx]
                    progress["completed_rounds"] = completed_rounds
                    self._save_progress(run_root, progress)
                else:
                    prev_report = saved_eval_report
                    prev_skills_dir = round_dir / "after_skills"
                    continue

            round_dir.mkdir(parents=True, exist_ok=True)

            # 1. Build training summary from previous round's eval sessions.
            train_summary = self.inner._build_reused_train_summary(prev_report)

            # 2. Run evolution.
            evolve_arm_root = round_dir / "evolve"
            evolution_summary = self._run_round_evolution(
                evolve_arm_root=evolve_arm_root,
                round_idx=round_idx,
                train_summary=train_summary,
                prev_cloud_dir=self._prev_cloud_dir(run_root, round_idx),
                prev_skills_dir=prev_skills_dir,
                task_specs=task_specs,
            )

            # 3. Materialize skills: copy prev_skills + pull newly evolved.
            after_skills_dir = self._materialize_round_skills(
                round_dir=round_dir,
                evolve_arm_root=evolve_arm_root,
                prev_skills_dir=prev_skills_dir,
            )

            # 4. Evaluate with the updated skills.
            eval_report = self._run_round_eval(
                run_root=run_root,
                round_dir=round_dir,
                task_specs=task_specs,
                skills_dir=after_skills_dir,
                round_idx=round_idx,
            )

            # 5. Save per-round report.
            self._save_round_report(
                round_dir=round_dir,
                round_idx=round_idx,
                train_summary=train_summary,
                evolution_summary=evolution_summary,
                eval_report=eval_report,
                baseline_report=baseline_report,
            )

            # 6. Update checkpoint.
            completed_rounds.append(round_idx)
            progress["completed_rounds"] = completed_rounds
            progress["last_completed_round"] = round_idx
            progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save_progress(run_root, progress)

            prev_report = eval_report
            prev_skills_dir = after_skills_dir

        # ---- Build final aggregated report ----
        report = self._build_iterative_report(run_root, task_specs, baseline_report)
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "report.md").write_text(self._build_markdown(report), encoding="utf-8")
        return report

    # ------------------------------------------------------------------
    # Round-level operations
    # ------------------------------------------------------------------

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
        """Run one evolution pass, accumulating skills from prior rounds."""
        evolve_arm_root.mkdir(parents=True, exist_ok=True)
        cloud_dir = evolve_arm_root / "cloud"
        cloud_dir.mkdir(parents=True, exist_ok=True)

        # Carry forward cumulative evolved skills from the previous round.
        if prev_cloud_dir is not None and prev_cloud_dir.exists():
            shutil.copytree(prev_cloud_dir, cloud_dir, dirs_exist_ok=True)

        group_id = f"{self.config.name}-iterative-round-{round_idx}"

        # Seed current skills into the cloud dir so the evolve server can
        # read ``current_skill`` content for each skill referenced by
        # sessions.  Without this, the evolve server would only see empty
        # skill slots and produce degraded edits.
        if prev_skills_dir.exists():
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
                evolve_aggregate_sessions=True,
                executor="replay",
            )
        )
        return runner._run_evolution(round_idx, train_summary=train_summary)

    def _materialize_round_skills(
        self,
        *,
        round_dir: Path,
        evolve_arm_root: Path,
        prev_skills_dir: Path,
    ) -> Path:
        """Copy previous skills and overlay only skills evolved this round.

        Instead of blindly pulling ALL skills from cloud (which may contain
        pre-existing artifacts from earlier phases), we read the
        ``evolve_skill_registry.json`` to determine which skills the evolve
        server actually touched, and only copy those.
        """
        after_skills_dir = round_dir / "after_skills"
        if after_skills_dir.exists():
            shutil.rmtree(after_skills_dir)
        after_skills_dir.mkdir(parents=True, exist_ok=True)

        if prev_skills_dir.exists():
            shutil.copytree(prev_skills_dir, after_skills_dir, dirs_exist_ok=True)

        cloud_dir = evolve_arm_root / "cloud"
        group_candidates = [d for d in cloud_dir.iterdir() if d.is_dir()] if cloud_dir.exists() else []
        for group_dir in group_candidates:
            registry_path = group_dir / "evolve_skill_registry.json"
            if not registry_path.exists():
                continue
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for skill_name in registry:
                src = group_dir / "skills" / skill_name / "SKILL.md"
                if not src.exists():
                    continue
                dst = after_skills_dir / skill_name / "SKILL.md"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)

        return after_skills_dir

    def _run_round_eval(
        self,
        *,
        run_root: Path,
        round_dir: Path,
        task_specs: list[dict[str, Any]],
        skills_dir: Path,
        round_idx: int,
    ) -> dict[str, Any]:
        """Evaluate all tasks using the given skills directory."""
        eval_root = round_dir / "eval"
        eval_root.mkdir(parents=True, exist_ok=True)

        # Check for existing eval report (checkpoint).
        existing_report = self._load_json(eval_root / "workspace" / "report.json")
        if existing_report is not None:
            return existing_report

        tasks_path = eval_root / "tasks.jsonl"
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

        rollouts = max(1, self.config.after_rollouts_per_task)
        arm_devices = self.inner._arm_device_count(task_specs, rollouts)
        # Port offset: baseline uses 0, round-1 uses 200, round-2 uses 400, etc.
        port_offset = round_idx * 200

        config_path = self.inner._write_arm_config(
            arm_root=eval_root,
            arm_name=f"iterative-round-{round_idx}-eval",
            tasks_path=tasks_path,
            initial_skills_dir=skills_dir,
            arm_devices=arm_devices,
            rollouts_per_task=rollouts,
            rounds=0,
            sync_enabled=False,
            evolve_enabled=False,
            eval_every_round=False,
            initial_eval_enabled=True,
            device_field="",
            port_offset=port_offset,
        )
        return self.inner._run_benchmark(config_path)

    # ------------------------------------------------------------------
    # Cloud dir chaining
    # ------------------------------------------------------------------

    def _prev_cloud_dir(self, run_root: Path, round_idx: int) -> Path | None:
        """Return the cloud dir from the previous round, if it exists."""
        if round_idx <= 1:
            # Round 1's evolution starts from scratch (or from the baseline arm's cloud
            # if it exists — but our baseline doesn't produce a cloud dir).
            return None
        prev = run_root / "arms" / f"round-{round_idx - 1}" / "evolve" / "cloud"
        return prev if prev.exists() else None

    # ------------------------------------------------------------------
    # Per-round report
    # ------------------------------------------------------------------

    def _save_round_report(
        self,
        *,
        round_dir: Path,
        round_idx: int,
        train_summary: dict[str, Any],
        evolution_summary: dict[str, Any],
        eval_report: dict[str, Any],
        baseline_report: dict[str, Any],
    ) -> None:
        eval_phase = eval_report.get("initial_eval", {}) or {}
        eval_summary = eval_report.get("summary") or {}
        eval_success = _safe_float(
            eval_summary.get("initial_eval_success_rate", eval_phase.get("success_rate"))
        )
        eval_score = _safe_float(
            eval_summary.get("initial_eval_mean_score", eval_phase.get("mean_score"))
        )

        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_summary = baseline_report.get("summary") or {}
        baseline_score = _safe_float(
            baseline_summary.get("initial_eval_mean_score", baseline_phase.get("mean_score"))
        )
        baseline_success = _safe_float(
            baseline_summary.get("initial_eval_success_rate", baseline_phase.get("success_rate"))
        )

        evolved_skills: list[str] = []
        manifest_path = str((evolution_summary.get("manifest_path")) or "")
        if manifest_path:
            evolved_skills = _load_manifest_names(manifest_path)

        round_report = {
            "round": round_idx,
            "eval_success_rate": eval_success,
            "eval_mean_score": eval_score,
            "score_gain_vs_baseline": round(eval_score - baseline_score, 6),
            "success_gain_vs_baseline": round(eval_success - baseline_success, 6),
            "skills_evolved_this_round": evolved_skills,
            "evolution_summary": evolution_summary.get("summary", {}),
            "train_summary": {
                "tasks": train_summary.get("tasks", 0),
                "successes": train_summary.get("successes", 0),
                "mean_score": train_summary.get("mean_score", 0.0),
            },
            "eval_report_path": str(round_dir / "eval" / "workspace" / "report.json"),
        }
        report_path = round_dir / "round_report.json"
        report_path.write_text(json.dumps(round_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Final aggregated report
    # ------------------------------------------------------------------

    def _build_iterative_report(
        self,
        run_root: Path,
        task_specs: list[dict[str, Any]],
        baseline_report: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_summary = baseline_report.get("summary") or {}
        baseline_score = _safe_float(
            baseline_summary.get("initial_eval_mean_score", baseline_phase.get("mean_score"))
        )
        baseline_success = _safe_float(
            baseline_summary.get("initial_eval_success_rate", baseline_phase.get("success_rate"))
        )

        trajectory_scores = [baseline_score]
        trajectory_successes = [baseline_success]
        cumulative_skills: list[list[str]] = [[]]
        all_skills_so_far: list[str] = []

        rounds_data: list[dict[str, Any]] = []
        for round_idx in range(1, self.config.iterative_rounds + 1):
            round_dir = run_root / "arms" / f"round-{round_idx}"
            round_report = self._load_json(round_dir / "round_report.json")
            if round_report is None:
                break

            round_score = _safe_float(round_report.get("eval_mean_score"))
            round_success = _safe_float(round_report.get("eval_success_rate"))

            new_skills = round_report.get("skills_evolved_this_round", [])
            all_skills_so_far = list(set(all_skills_so_far) | set(new_skills))

            prev_score = trajectory_scores[-1]
            rounds_data.append({
                "round": round_idx,
                "eval_mean_score": round_score,
                "eval_success_rate": round_success,
                "score_gain_vs_baseline": round(round_score - baseline_score, 6),
                "score_gain_vs_prev": round(round_score - prev_score, 6),
                "success_gain_vs_baseline": round(round_success - baseline_success, 6),
                "skills_evolved_this_round": new_skills,
                "cumulative_skills": list(all_skills_so_far),
            })

            trajectory_scores.append(round_score)
            trajectory_successes.append(round_success)
            cumulative_skills.append(list(all_skills_so_far))

        # Per-query comparison: baseline vs last round.
        per_query = self._build_per_query_comparison(
            run_root, task_specs, baseline_report, len(rounds_data)
        )

        last_score = trajectory_scores[-1]
        last_success = trajectory_successes[-1]

        return {
            "name": self.config.name,
            "run_root": str(run_root),
            "benchmark_root": str(self.inner.benchmark_root),
            "agent_model": self.config.cluster_llm_model_id,
            "evolve_model": self.config.evolve_model,
            "iterative_rounds": self.config.iterative_rounds,
            "rounds_completed": len(rounds_data),
            "baseline": {
                "success_rate": baseline_success,
                "mean_score": baseline_score,
            },
            "final": {
                "success_rate": last_success,
                "mean_score": last_score,
                "score_gain_vs_baseline": round(last_score - baseline_score, 6),
                "success_gain_vs_baseline": round(last_success - baseline_success, 6),
            },
            "rounds": rounds_data,
            "per_query": per_query,
            "trajectory": {
                "mean_scores": trajectory_scores,
                "success_rates": trajectory_successes,
                "cumulative_skills": cumulative_skills,
            },
        }

    def _build_per_query_comparison(
        self,
        run_root: Path,
        task_specs: list[dict[str, Any]],
        baseline_report: dict[str, Any],
        last_round: int,
    ) -> list[dict[str, Any]]:
        """Build per-task comparison between baseline and the last completed round."""
        rollouts = max(1, self.config.after_rollouts_per_task)

        baseline_phase = baseline_report.get("initial_eval", {}) or {}
        baseline_records = _limit_records_by_task_rollouts(
            _flatten_phase_records(baseline_phase),
            max(1, self.config.baseline_rollouts_per_task),
        )
        before_by_task = self.inner._group_records_by_task(baseline_records)

        after_by_task: dict[str, dict[str, Any]] = {}
        if last_round > 0:
            last_eval_report = self._load_json(
                run_root / "arms" / f"round-{last_round}" / "eval" / "workspace" / "report.json"
            )
            if last_eval_report:
                after_phase = last_eval_report.get("initial_eval", {}) or {}
                after_records = _limit_records_by_task_rollouts(
                    _flatten_phase_records(after_phase),
                    rollouts,
                )
                after_by_task = self.inner._group_records_by_task(after_records)

        per_query: list[dict[str, Any]] = []
        for spec in task_specs:
            task_id = spec["task_id"]
            before = before_by_task.get(task_id, {})
            after = after_by_task.get(task_id, {})
            before_score = _safe_float(before.get("score"))
            after_score = _safe_float(after.get("score"))
            per_query.append({
                "task_id": task_id,
                "category": spec.get("category", ""),
                "before": before,
                "after": after,
                "score_gain": round(after_score - before_score, 6),
                "improved": after_score > before_score,
            })
        return per_query

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def _build_markdown(self, report: dict[str, Any]) -> str:
        baseline = report.get("baseline", {})
        final = report.get("final", {})
        rounds = report.get("rounds", [])

        lines = [
            f"# {report.get('name', 'Iterative Evolve Experiment')}",
            "",
            "## Summary",
            "",
            f"- Agent model: `{report.get('agent_model', 'unknown')}`",
            f"- Evolve model: `{report.get('evolve_model', 'unknown')}`",
            f"- Total rounds: {report.get('iterative_rounds', 0)}",
            f"- Rounds completed: {report.get('rounds_completed', 0)}",
            f"- Baseline mean score: {baseline.get('mean_score', 0):.3f}",
            f"- Final mean score: {final.get('mean_score', 0):.3f}",
            f"- Overall score gain: {final.get('score_gain_vs_baseline', 0):+.3f}",
            f"- Baseline success rate: {baseline.get('success_rate', 0):.3f}",
            f"- Final success rate: {final.get('success_rate', 0):.3f}",
            f"- Overall success gain: {final.get('success_gain_vs_baseline', 0):+.3f}",
            "",
            "## Round Trajectory",
            "",
            "| Round | Mean Score | Score vs Baseline | Score vs Prev | Success Rate | New Skills |",
            "|-------|-----------|-------------------|---------------|-------------|------------|",
            f"| 0 (baseline) | {baseline.get('mean_score', 0):.3f} | -- | -- | {baseline.get('success_rate', 0):.3f} | -- |",
        ]
        for rd in rounds:
            skills_str = ", ".join(rd.get("skills_evolved_this_round", [])) or "(none)"
            lines.append(
                f"| {rd['round']} | {rd.get('eval_mean_score', 0):.3f} "
                f"| {rd.get('score_gain_vs_baseline', 0):+.3f} "
                f"| {rd.get('score_gain_vs_prev', 0):+.3f} "
                f"| {rd.get('eval_success_rate', 0):.3f} "
                f"| {skills_str} |"
            )

        lines.extend([
            "",
            "## Per Query (Baseline vs Final Round)",
            "",
            "| Task | Before | After | Gain |",
            "|------|--------|-------|------|",
        ])
        for item in report.get("per_query", []):
            lines.append(
                "| {task} | {before:.3f} | {after:.3f} | {gain:+.3f} |".format(
                    task=item["task_id"],
                    before=_safe_float((item.get("before") or {}).get("score")),
                    after=_safe_float((item.get("after") or {}).get("score")),
                    gain=_safe_float(item.get("score_gain")),
                )
            )

        lines.extend([
            "",
            "## Artifacts",
            "",
            f"- Run root: `{report.get('run_root', '.')}`",
            f"- JSON report: `{report.get('run_root', '.')}/report.json`",
        ])
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Import from previous batch_evolve run
    # ------------------------------------------------------------------

    def _import_from_batch_evolve(self, source_run_root: Path) -> Path:
        """Import a previous batch_evolve run as round-1 of a new iterative run.

        Layout mapping:
            source: arms/baseline/              -> new: arms/baseline/          (symlink)
            source: initial_skills/             -> new: initial_skills/         (symlink)
            source: selected_tasks.json         -> new: selected_tasks.json     (copy)
            source: arms/batch_evolve/cloud/    -> new: arms/round-1/evolve/cloud/  (symlink)
            source: arms/batch_evolve/workspace/-> new: arms/round-1/evolve/workspace/ (symlink)
            source: arms/batch_evolve/after_skills/ -> new: arms/round-1/after_skills/ (symlink)
            source: arms/batch_evolve/workspace/report.json -> new: arms/round-1/eval/workspace/report.json (symlink)
            source: arms/batch_evolve/wildclawbench_output/ -> new: arms/round-1/eval/wildclawbench_output/ (symlink)
        """
        if not source_run_root.exists():
            raise FileNotFoundError(f"import source not found: {source_run_root}")

        # Validate source has the expected structure.
        source_baseline = source_run_root / "arms" / "baseline"
        source_batch = source_run_root / "arms" / "batch_evolve"
        source_baseline_report = source_baseline / "workspace" / "report.json"
        source_after_report = source_batch / "workspace" / "report.json"
        for required in (source_baseline, source_batch, source_baseline_report, source_after_report):
            if not required.exists():
                raise FileNotFoundError(
                    f"import source missing required path: {required}"
                )

        # Create new run_root.
        run_root = self.inner._resolve_run_root()

        # Check if import was already done.
        progress = self._load_progress(run_root)
        if progress.get("imported_from") == str(source_run_root):
            return run_root

        # Remove stale report.json if present (so we don't short-circuit).
        stale_report = run_root / "report.json"
        if stale_report.exists():
            stale_report.unlink()

        # Copy selected_tasks.json.
        source_tasks = source_run_root / "selected_tasks.json"
        if source_tasks.exists() and not (run_root / "selected_tasks.json").exists():
            shutil.copy2(source_tasks, run_root / "selected_tasks.json")

        # Symlink initial_skills.
        dst_initial = run_root / "initial_skills"
        src_initial = source_run_root / "initial_skills"
        if src_initial.exists() and not dst_initial.exists():
            dst_initial.symlink_to(src_initial)

        # Symlink baseline arm.
        dst_baseline = run_root / "arms" / "baseline"
        dst_baseline.parent.mkdir(parents=True, exist_ok=True)
        if not dst_baseline.exists():
            dst_baseline.symlink_to(source_baseline)

        # Build round-1 directory from batch_evolve arm.
        round_dir = run_root / "arms" / "round-1"
        round_dir.mkdir(parents=True, exist_ok=True)

        # round-1/evolve/ — link the cloud dir and workspace from batch_evolve.
        evolve_dir = round_dir / "evolve"
        evolve_dir.mkdir(parents=True, exist_ok=True)
        src_cloud = source_batch / "cloud"
        dst_cloud = evolve_dir / "cloud"
        if src_cloud.exists() and not dst_cloud.exists():
            dst_cloud.symlink_to(src_cloud)
        src_evolve_ws = source_batch / "workspace"
        dst_evolve_ws = evolve_dir / "workspace"
        if src_evolve_ws.exists() and not dst_evolve_ws.exists():
            dst_evolve_ws.symlink_to(src_evolve_ws)

        # round-1/after_skills/ — link after_skills.
        src_after_skills = source_batch / "after_skills"
        dst_after_skills = round_dir / "after_skills"
        if src_after_skills.exists() and not dst_after_skills.exists():
            dst_after_skills.symlink_to(src_after_skills)

        # round-1/eval/ — create eval dir pointing to the batch_evolve workspace report.
        eval_dir = round_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        dst_eval_ws = eval_dir / "workspace"
        if not dst_eval_ws.exists():
            dst_eval_ws.symlink_to(src_evolve_ws)
        src_wb_output = source_batch / "wildclawbench_output"
        dst_wb_output = eval_dir / "wildclawbench_output"
        if src_wb_output.exists() and not dst_wb_output.exists():
            dst_wb_output.symlink_to(src_wb_output)

        # Build round-1 report from the source data.
        baseline_report = self._load_json(source_baseline_report)
        after_report = self._load_json(source_after_report)
        if baseline_report and after_report:
            # Find the manifest.jsonl from the source cloud dir to identify evolved skills.
            source_group_id = f"{self._guess_source_group_id(source_batch / 'cloud')}"
            manifest_path = source_batch / "cloud" / source_group_id / "manifest.jsonl"
            evolution_summary = {
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                "summary": {},
            }
            self._save_round_report(
                round_dir=round_dir,
                round_idx=1,
                train_summary=self.inner._build_reused_train_summary(baseline_report),
                evolution_summary=evolution_summary,
                eval_report=after_report,
                baseline_report=baseline_report,
            )

        # Write progress with round-1 already completed.
        progress = {
            "config_name": self.config.name,
            "total_rounds": self.config.iterative_rounds,
            "baseline_done": True,
            "baseline_report_path": str(run_root / "arms" / "baseline" / "workspace" / "report.json"),
            "completed_rounds": [1],
            "last_completed_round": 1,
            "imported_from": str(source_run_root),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_progress(run_root, progress)

        return run_root

    @staticmethod
    def _guess_source_group_id(cloud_dir: Path) -> str:
        """Return the first subdirectory name under cloud_dir as the group_id."""
        if cloud_dir.exists():
            for child in cloud_dir.iterdir():
                if child.is_dir():
                    return child.name
        return ""

    # ------------------------------------------------------------------
    # Progress / checkpoint helpers
    # ------------------------------------------------------------------

    def _load_progress(self, run_root: Path) -> dict[str, Any]:
        path = run_root / "iterative_progress.json"
        data = self._load_json(path)
        if isinstance(data, dict):
            return data
        return {
            "config_name": self.config.name,
            "total_rounds": self.config.iterative_rounds,
            "baseline_done": False,
            "completed_rounds": [],
            "last_completed_round": 0,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _save_progress(self, run_root: Path, progress: dict[str, Any]) -> None:
        progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path = run_root / "iterative_progress.json"
        path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

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


# ------------------------------------------------------------------
# Config loader / entry point
# ------------------------------------------------------------------

def run_wildclawbench_iterative_evolve_from_config(config_path: str) -> dict[str, Any]:
    """Load config and run the iterative evolve experiment."""
    config = WildClawBenchBatchEvolveExperimentConfig.from_file(config_path)
    runner = IterativeEvolveRunner(config)
    return runner.run()
