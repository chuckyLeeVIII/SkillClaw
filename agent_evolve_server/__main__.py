"""
CLI entry point — ``python -m agent_evolve_server``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .config import AgentEvolveServerConfig
from .server import AgentEvolveServer

logger = logging.getLogger("agent_evolve_server")


def _build_config_from_args(args: argparse.Namespace) -> AgentEvolveServerConfig:
    """Merge CLI args, environment variables, and optionally skillclaw config."""
    config = AgentEvolveServerConfig.from_env()

    if args.use_skillclaw_config:
        try:
            from skillclaw.config_store import ConfigStore

            sc_config = ConfigStore().to_skillclaw_config()
            config = AgentEvolveServerConfig.from_skillclaw_config(sc_config)
        except Exception as e:
            logger.warning("Could not load skillclaw config: %s — falling back to env", e)

    # Storage overrides
    if args.storage_backend:
        config.storage_backend = args.storage_backend
    if args.storage_endpoint or args.oss_endpoint:
        config.storage_endpoint = args.storage_endpoint or args.oss_endpoint
        config.oss_endpoint = config.storage_endpoint
    if args.storage_bucket or args.oss_bucket:
        config.storage_bucket = args.storage_bucket or args.oss_bucket
        config.oss_bucket = config.storage_bucket
    if args.storage_region:
        config.storage_region = args.storage_region
    if args.local_root:
        config.local_root = args.local_root
        if not config.storage_backend:
            config.storage_backend = "local"
    if not config.storage_backend:
        if args.oss_endpoint or args.oss_bucket:
            config.storage_backend = "oss"
        elif config.storage_bucket or config.storage_endpoint:
            config.storage_backend = "s3"

    # General overrides
    if args.group_id:
        config.group_id = args.group_id
    if args.model:
        config.llm_model = args.model
    if args.interval:
        config.interval_seconds = args.interval
    if args.port:
        config.http_port = args.port

    # OpenClaw-specific overrides
    if args.openclaw_bin:
        config.openclaw_bin = args.openclaw_bin
    if args.openclaw_home:
        config.openclaw_home = args.openclaw_home
    if args.agent_timeout:
        config.agent_timeout = args.agent_timeout
    if args.workspace_root:
        config.workspace_root = args.workspace_root
    if args.agents_md:
        config.agents_md_path = args.agents_md
    if args.llm_api_type:
        config.llm_api_type = args.llm_api_type

    # --fresh / --no-fresh
    if args.fresh is not None:
        config.fresh = args.fresh

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillClaw Agent Evolve Server (OpenClaw-driven)",
    )

    # --- Execution modes --- #
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use local mock/ directory instead of remote object storage",
    )
    parser.add_argument("--mock-root", type=str, default=None,
                        help="Custom root directory for mock mode")
    parser.add_argument("--port", type=int, default=None,
                        help="HTTP trigger port (enables HTTP server)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Periodic interval in seconds")

    # --- Model / storage --- #
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    parser.add_argument("--group-id", type=str, default=None, help="Shared storage group ID")
    parser.add_argument("--storage-backend", type=str, default=None)
    parser.add_argument("--storage-endpoint", type=str, default=None)
    parser.add_argument("--storage-bucket", type=str, default=None)
    parser.add_argument("--storage-region", type=str, default=None)
    parser.add_argument("--oss-endpoint", type=str, default=None)
    parser.add_argument("--oss-bucket", type=str, default=None)
    parser.add_argument("--local-root", type=str, default=None,
                        help="Use a local directory as the evolve backend root")
    parser.add_argument("--use-skillclaw-config", action="store_true",
                        help="Load settings from skillclaw's config store")

    # --- OpenClaw-specific --- #
    parser.add_argument("--openclaw-bin", type=str, default=None,
                        help="Path to openclaw executable (default: 'openclaw')")
    parser.add_argument("--openclaw-home", type=str, default=None,
                        help="OPENCLAW_HOME directory for agent state")
    fresh_group = parser.add_mutually_exclusive_group()
    fresh_group.add_argument("--fresh", dest="fresh", action="store_true", default=None,
                             help="Wipe agent state each cycle (no memory)")
    fresh_group.add_argument("--no-fresh", dest="fresh", action="store_false",
                             help="Preserve agent state across cycles (multi-round memory)")
    parser.add_argument("--agent-timeout", type=int, default=None,
                        help="Agent execution timeout in seconds (default: 600)")
    parser.add_argument("--workspace-root", type=str, default=None,
                        help="Workspace directory for agent file operations")
    parser.add_argument("--agents-md", type=str, default=None,
                        help="Custom AGENTS.md path (overrides built-in template)")
    parser.add_argument("--llm-api-type", type=str, default=None,
                        choices=["openai-completions", "anthropic-messages",
                                 "openai-responses", "google-generative-ai", "ollama"],
                        help="LLM provider API type (default: anthropic-messages)")

    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    config = _build_config_from_args(args)

    # Validate storage config (unless mock mode)
    if not args.mock:
        backend = (config.storage_backend or "").strip().lower()
        if backend == "local":
            if not config.local_root:
                logger.error("Local storage backend requires --local-root or EVOLVE_STORAGE_LOCAL_ROOT.")
                raise SystemExit(1)
        elif backend == "oss":
            if not config.storage_endpoint or not config.storage_bucket:
                logger.error(
                    "OSS backend requires endpoint and bucket. "
                    "Set EVOLVE_STORAGE_ENDPOINT / EVOLVE_STORAGE_BUCKET, "
                    "or use --use-skillclaw-config."
                )
                raise SystemExit(1)
        elif backend:
            if not config.storage_bucket:
                logger.error(
                    "Storage bucket is required for remote backends. "
                    "Use --local-root for local mode, or use --mock."
                )
                raise SystemExit(1)

    server = AgentEvolveServer(config, mock=args.mock, mock_root=args.mock_root)

    if args.once or args.mock:
        summary = asyncio.run(server.run_once())
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.port is not None:
        import uvicorn

        app = server.create_http_app()

        async def _run_with_http():
            uv_config = uvicorn.Config(
                app, host="0.0.0.0", port=config.http_port, log_level="info",
            )
            uv_server = uvicorn.Server(uv_config)
            await asyncio.gather(server.run_periodic(), uv_server.serve())

        asyncio.run(_run_with_http())
    else:
        asyncio.run(server.run_periodic())


if __name__ == "__main__":
    main()
