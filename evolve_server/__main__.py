"""
CLI entry point — ``python -m evolve_server``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .config import EvolveServerConfig
from .server import EvolveServer

logger = logging.getLogger("evolve_server")


def _build_config_from_args(args: argparse.Namespace) -> EvolveServerConfig:
    """Merge CLI args, environment variables, and optionally skillclaw config."""
    config = EvolveServerConfig.from_env()

    if args.use_skillclaw_config:
        try:
            from skillclaw.config_store import ConfigStore

            sc_config = ConfigStore().to_skillclaw_config()
            config = EvolveServerConfig.from_skillclaw_config(sc_config)
        except Exception as e:
            logger.warning("Could not load skillclaw config: %s — falling back to env", e)

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
    if args.group_id:
        config.group_id = args.group_id
    if args.model:
        config.llm_model = args.model
    if args.interval:
        config.interval_seconds = args.interval
    if args.port:
        config.http_port = args.port
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillClaw Evolve Server")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--mock", action="store_true",
                        help="Use local mock/ directory instead of remote object storage")
    parser.add_argument("--mock-root", type=str, default=None,
                        help="Custom root directory for mock mode")
    parser.add_argument("--port", type=int, default=None,
                        help="HTTP trigger port (enables HTTP server)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Periodic interval in seconds")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    parser.add_argument("--group-id", type=str, default=None, help="Shared storage group ID")
    parser.add_argument("--storage-backend", type=str, default=None, help="Storage backend: local, s3, or oss")
    parser.add_argument("--storage-endpoint", type=str, default=None)
    parser.add_argument("--storage-bucket", type=str, default=None)
    parser.add_argument("--storage-region", type=str, default=None)
    parser.add_argument("--oss-endpoint", type=str, default=None)
    parser.add_argument("--oss-bucket", type=str, default=None)
    parser.add_argument(
        "--local-root",
        type=str,
        default=None,
        help="Use a local directory as the evolve backend root",
    )
    parser.add_argument("--use-skillclaw-config", action="store_true",
                        help="Load shared storage and LLM settings from skillclaw's config store")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    config = _build_config_from_args(args)

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
                    "Set EVOLVE_STORAGE_ENDPOINT / EVOLVE_STORAGE_BUCKET, use legacy EVOLVE_OSS_* vars, "
                    "or use --use-skillclaw-config."
                )
                raise SystemExit(1)
        else:
            if not config.storage_bucket:
                logger.error(
                    "Storage bucket is required for remote backends. "
                    "Set EVOLVE_STORAGE_BUCKET, use legacy EVOLVE_OSS_BUCKET, "
                    "use --use-skillclaw-config, use --local-root for local mode, or use --mock."
                )
                raise SystemExit(1)

    server = EvolveServer(config, mock=args.mock, mock_root=args.mock_root)

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
