"""
Configuration dataclass for the Evolve Server.

On import, automatically loads ``evolve_server/.env`` (if present) via
``python-dotenv`` so that all config values can live in a single ``.env``
file rather than being exported in the shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Best-effort load of the ``.env`` next to this file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


_load_dotenv()


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _infer_storage_backend(endpoint: str, bucket: str, local_root: str) -> str:
    backend = _first_env("EVOLVE_STORAGE_BACKEND", default="").strip().lower()
    if backend:
        return backend
    if local_root:
        return "local"
    if any(os.environ.get(name) for name in ("EVOLVE_OSS_ENDPOINT", "EVOLVE_OSS_BUCKET", "EVOLVE_OSS_KEY_ID", "EVOLVE_OSS_KEY_SECRET")):
        return "oss"
    if endpoint or bucket:
        return "s3"
    return ""


@dataclass
class EvolveServerConfig:
    # Storage
    storage_backend: str = ""
    storage_endpoint: str = ""
    storage_bucket: str = ""
    storage_access_key_id: str = ""
    storage_secret_access_key: str = ""
    storage_region: str = ""
    storage_session_token: str = ""

    # Backward-compatible aliases for OSS-only integrations.
    oss_endpoint: str = ""
    oss_bucket: str = ""
    oss_access_key_id: str = ""
    oss_access_key_secret: str = ""
    group_id: str = "default"
    local_root: str = ""

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 100000
    llm_temperature: float = 0.4
    evolve_strategy: str = "dynamic_edit_conservative"
    use_success_feedback: bool = True

    # Evolution
    evolve_batch_size: int = 20
    reject_rewrite: bool = False  # Reject skill improvements that look like full rewrites
    use_session_judge: bool = True
    debug_dump_dir: str = ""

    # Scheduling
    interval_seconds: int = 600
    http_port: int = 8787

    # Local persistence
    history_path: str = "evolve_history.jsonl"
    processed_log_path: str = "evolve_processed.json"

    @classmethod
    def from_env(cls) -> "EvolveServerConfig":
        """Populate every field from environment variables.

        The ``.env`` file has already been loaded into ``os.environ`` by
        ``_load_dotenv()`` at module-import time, so a plain
        ``os.environ.get`` picks up both shell exports and ``.env`` values.
        """
        storage_endpoint = _first_env("EVOLVE_STORAGE_ENDPOINT", "EVOLVE_OSS_ENDPOINT")
        storage_bucket = _first_env("EVOLVE_STORAGE_BUCKET", "EVOLVE_OSS_BUCKET")
        storage_access_key_id = _first_env("EVOLVE_STORAGE_ACCESS_KEY_ID", "EVOLVE_OSS_KEY_ID")
        storage_secret_access_key = _first_env("EVOLVE_STORAGE_SECRET_ACCESS_KEY", "EVOLVE_OSS_KEY_SECRET")
        storage_region = _first_env("EVOLVE_STORAGE_REGION")
        storage_session_token = _first_env("EVOLVE_STORAGE_SESSION_TOKEN")
        local_root = _first_env("EVOLVE_STORAGE_LOCAL_ROOT", "EVOLVE_LOCAL_ROOT")
        storage_backend = _infer_storage_backend(storage_endpoint, storage_bucket, local_root)
        return cls(
            storage_backend=storage_backend,
            storage_endpoint=storage_endpoint,
            storage_bucket=storage_bucket,
            storage_access_key_id=storage_access_key_id,
            storage_secret_access_key=storage_secret_access_key,
            storage_region=storage_region,
            storage_session_token=storage_session_token,
            oss_endpoint=storage_endpoint,
            oss_bucket=storage_bucket,
            oss_access_key_id=storage_access_key_id,
            oss_access_key_secret=storage_secret_access_key,
            group_id=os.environ.get("EVOLVE_GROUP_ID", "default"),
            local_root=local_root,
            llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.environ.get("EVOLVE_MODEL", "gpt-4o"),
            llm_max_tokens=int(os.environ.get("EVOLVE_LLM_MAX_TOKENS", "100000")),
            llm_temperature=float(os.environ.get("EVOLVE_LLM_TEMPERATURE", "0.4")),
            evolve_strategy=os.environ.get("EVOLVE_STRATEGY", "dynamic_edit_conservative"),
            use_success_feedback=os.environ.get("EVOLVE_USE_SUCCESS_FEEDBACK", "1").lower() not in {"0", "false", "no"},
            evolve_batch_size=int(os.environ.get("EVOLVE_BATCH_SIZE", "20")),
            reject_rewrite=os.environ.get("EVOLVE_REJECT_REWRITE", "0").lower() in {"1", "true", "yes"},
            use_session_judge=os.environ.get("EVOLVE_USE_SESSION_JUDGE", "1").lower() not in {"0", "false", "no"},
            interval_seconds=int(os.environ.get("EVOLVE_INTERVAL", "600")),
            http_port=int(os.environ.get("EVOLVE_PORT", "8787")),
            history_path=os.environ.get("EVOLVE_HISTORY_LOG", "evolve_history.jsonl"),
            processed_log_path=os.environ.get("EVOLVE_PROCESSED_LOG", "evolve_processed.json"),
        )

    @classmethod
    def from_skillclaw_config(cls, config) -> "EvolveServerConfig":
        """Build from an existing ``SkillClawConfig`` (reuse sharing + LLM settings)."""
        storage_backend = str(getattr(config, "sharing_backend", "") or "").strip().lower()
        storage_endpoint = str(
            getattr(config, "sharing_endpoint", "")
            or getattr(config, "sharing_oss_endpoint", "")
            or ""
        )
        storage_bucket = str(
            getattr(config, "sharing_bucket", "")
            or getattr(config, "sharing_oss_bucket", "")
            or ""
        )
        storage_access_key_id = str(
            getattr(config, "sharing_access_key_id", "")
            or getattr(config, "sharing_oss_access_key_id", "")
            or ""
        )
        storage_secret_access_key = str(
            getattr(config, "sharing_secret_access_key", "")
            or getattr(config, "sharing_oss_access_key_secret", "")
            or ""
        )
        local_root = str(getattr(config, "sharing_local_root", "") or os.environ.get("EVOLVE_LOCAL_ROOT", ""))
        return cls(
            storage_backend=storage_backend or ("local" if local_root else "s3" if (storage_bucket or storage_endpoint) else "oss"),
            storage_endpoint=storage_endpoint,
            storage_bucket=storage_bucket,
            storage_access_key_id=storage_access_key_id,
            storage_secret_access_key=storage_secret_access_key,
            storage_region=str(getattr(config, "sharing_region", "") or ""),
            storage_session_token=str(getattr(config, "sharing_session_token", "") or ""),
            oss_endpoint=storage_endpoint,
            oss_bucket=storage_bucket,
            oss_access_key_id=storage_access_key_id,
            oss_access_key_secret=storage_secret_access_key,
            group_id=config.sharing_group_id,
            local_root=local_root,
            llm_api_key=config.llm_api_key or config.prm_api_key,
            llm_base_url=config.llm_api_base or config.prm_url,
            llm_model=os.environ.get("EVOLVE_MODEL", config.llm_model_id or "gpt-4o"),
            evolve_strategy=os.environ.get("EVOLVE_STRATEGY", "dynamic_edit_conservative"),
            use_success_feedback=os.environ.get("EVOLVE_USE_SUCCESS_FEEDBACK", "1").lower() not in {"0", "false", "no"},
            evolve_batch_size=int(os.environ.get("EVOLVE_BATCH_SIZE", "20")),
            reject_rewrite=os.environ.get("EVOLVE_REJECT_REWRITE", "0").lower() in {"1", "true", "yes"},
            use_session_judge=os.environ.get("EVOLVE_USE_SESSION_JUDGE", "1").lower() not in {"0", "false", "no"},
        )
