"""
Local skill hub for benchmark and offline group-sync experiments.

This mirrors the public shape of SkillHub but uses a shared filesystem
directory instead of OSS. It is useful for:

- multi-device benchmark simulation on a single machine
- local development without cloud credentials
- deterministic integration tests for skill push/pull logic
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..skill_hub import _compute_sha256

logger = logging.getLogger(__name__)


class LocalSkillHub:
    """Sync skills through a shared local directory."""

    def __init__(
        self,
        root_dir: str,
        group_id: str = "default",
        user_alias: str = "",
    ):
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._group_id = group_id
        self._user_alias = user_alias or os.environ.get("USER", "anonymous")
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _group_dir(self) -> Path:
        return self._root_dir / self._group_id

    def _manifest_path(self) -> Path:
        return self._group_dir() / "manifest.jsonl"

    def _skill_path(self, skill_name: str) -> Path:
        return self._group_dir() / "skills" / skill_name / "SKILL.md"

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        path = self._manifest_path()
        if not path.exists():
            return {}

        manifest: dict[str, dict[str, Any]] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning("[LocalSkillHub] failed to read manifest: %s", e)
            return {}

        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("name")
            if name:
                manifest[str(name)] = rec
        return manifest

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(rec, ensure_ascii=False) for rec in manifest.values()]
        content = "\n".join(lines) + ("\n" if lines else "")
        path.write_text(content, encoding="utf-8")

    def push_skills(
        self,
        skills_dir: str,
        skill_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        import glob

        paths = sorted(glob.glob(os.path.join(skills_dir, "*", "SKILL.md")))
        if not paths:
            return {"uploaded": 0, "skipped": 0, "filtered": 0, "total_local": 0}

        manifest = self._load_manifest()
        uploaded = 0
        skipped = 0
        filtered = 0

        stats = (skill_filter or {}).get("stats", {})
        min_inj = (skill_filter or {}).get("min_injections", 0)
        min_eff = (skill_filter or {}).get("min_effectiveness", 0.0)
        use_filter = skill_filter is not None

        for path_str in paths:
            path = Path(path_str)
            skill_name = path.parent.name

            if use_filter and skill_name in stats:
                entry = stats[skill_name]
                inj = entry.get("inject_count", 0)
                eff = entry.get("effectiveness", 0.5)
                if inj >= min_inj and eff < min_eff:
                    filtered += 1
                    continue

            local_sha = _compute_sha256(str(path))
            remote_rec = manifest.get(skill_name)
            if remote_rec and remote_rec.get("sha256") == local_sha:
                skipped += 1
                continue

            target = self._skill_path(skill_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

            manifest[skill_name] = {
                "name": skill_name,
                "sha256": local_sha,
                "uploaded_by": self._user_alias,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            uploaded += 1

        if uploaded > 0:
            self._save_manifest(manifest)

        return {"uploaded": uploaded, "skipped": skipped, "filtered": filtered, "total_local": len(paths)}

    def pull_skills(self, skills_dir: str) -> dict[str, int]:
        manifest = self._load_manifest()
        if not manifest:
            return {"downloaded": 0, "skipped": 0, "total_remote": 0}

        skills_root = Path(skills_dir).expanduser().resolve()
        skills_root.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        skipped = 0
        for skill_name, rec in manifest.items():
            src = self._skill_path(skill_name)
            dst = skills_root / skill_name / "SKILL.md"
            if not src.exists():
                continue
            if dst.exists() and _compute_sha256(str(dst)) == rec.get("sha256"):
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            downloaded += 1

        return {"downloaded": downloaded, "skipped": skipped, "total_remote": len(manifest)}

    def sync_skills(
        self,
        skills_dir: str,
        skill_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, dict[str, int]]:
        pull = self.pull_skills(skills_dir)
        push = self.push_skills(skills_dir, skill_filter=skill_filter)
        return {"pull": pull, "push": push}

    def list_remote(self) -> list[dict[str, Any]]:
        return list(self._load_manifest().values())
