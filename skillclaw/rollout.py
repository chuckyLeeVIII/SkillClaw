# Adapted from MetaClaw
"""
Rollout worker for SkillClaw.

Hosts the SkillClawAPIServer in a background thread and provides the
output queue for collecting ConversationSamples.
"""

import asyncio
import atexit
import queue
import threading

from .api_server import SkillClawAPIServer
from .config import SkillClawConfig

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(
    config: SkillClawConfig,
    sampling_client=None,
    skill_manager=None,
    prm_scorer=None,
):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(
                config, sampling_client, skill_manager, prm_scorer,
            )
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    def __init__(
        self,
        config: SkillClawConfig,
        sampling_client=None,
        skill_manager=None,
        prm_scorer=None,
        last_request_tracker=None,
    ):
        self.config = config
        self.running = True
        self.output_queue = queue.Queue(maxsize=100000)
        self.worker_thread = None
        self._submission_enabled = threading.Event()
        self._server = SkillClawAPIServer(
            config=config,
            output_queue=self.output_queue,
            submission_enabled=self._submission_enabled,
            sampling_client=sampling_client,
            skill_manager=skill_manager,
            prm_scorer=prm_scorer,
            last_request_tracker=last_request_tracker,
        )

    async def continuous_worker_loop(self):
        while self.running:
            await asyncio.sleep(1.0)

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        self._server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self.worker_thread_func, daemon=True
            )
            self.worker_thread.start()

    def wait_until_ready(self, timeout_s: float = 30.0) -> bool:
        return self._server.wait_until_ready(timeout_s=timeout_s)

    def stop(self):
        self.running = False
        self._submission_enabled.clear()
        self._server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            print("[RolloutWorker] submission paused")

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            print("[RolloutWorker] submission resumed")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


atexit.register(stop_global_worker)
