from __future__ import annotations

from disco.orchestrator import Orchestrator

"""Kubernetes-friendly Disco server supervisor.

Spawns multiple Worker processes (optionally an orchestrator placeholder) and
ensures **one ZooKeeper/Kazoo client per OS process** by creating Cluster clients
inside each process via Cluster.make_cluster(...).

Design goals:
- Multi-worker per pod (shared IPC queues).
- PID 1 friendliness: handle SIGTERM/SIGINT, reap children, enforce shutdown.
- Workers are identified by their Cluster address string "host:port".
"""

import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing import cpu_count
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MPQueue
from typing import Dict, List, Mapping, Optional, Sequence, Any, cast
import logging
from tools import mp_logging

from disco.cluster import Cluster
from disco.config import AppSettings, ConfigError
from disco.worker import Worker

logger = mp_logging.getLogger(__name__)  # optional, but handy


_SINGLETON_LOCK = threading.Lock()
_SERVER_RUNNING = False


def _parse_log_level(level: str) -> int:
    mapping = logging.getLevelNamesMapping()
    lvl = mapping.get(level.upper())
    if isinstance(lvl, int):
        return lvl
    raise ConfigError(f"Invalid logging.level={level!r}. Expected one of: "
                      f"{sorted({k for k,v in mapping.items() if isinstance(v,int) and isinstance(k,str)})}")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    address: str
    name: str


def _is_loopback_host(host: str) -> bool:
    h = host.strip().lower()
    if h in {"localhost", "localhost.localdomain"}:
        return True
    if h.startswith("127."):
        return True
    if h == "::1":
        return True
    return False


def _infer_non_loopback_ipv4() -> Optional[str]:
    """Infer a non-loopback IPv4 address without relying on env.

    Uses a UDP "connect" trick to let the OS select an outbound interface.
    This does not need the remote endpoint to be reachable and sends no packets.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = cast(str, s.getsockname()[0])
            if ip and not ip.startswith("127."):
                return ip
        finally:
            s.close()
    except OSError:
        pass

    # Fallback: hostname resolution (often returns loopback in containers).
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass

    return None


def _resolve_bind_host(settings: AppSettings, explicit: Optional[str]) -> str:
    """Resolve a bind/advertise host suitable for worker addresses.

    Rules:
    - If `explicit` is provided: accept as-is (including localhost) for dev.
    - Else prefer settings.grpc.bind_host if present.
    - Else infer a non-loopback IP.
    - Never default to localhost/127.* automatically.
    """
    if explicit is not None:
        return explicit

    # Settings-driven host (via DISCO_GRPC__BIND_HOST)
    bind_host = settings.grpc.bind_host  # recommended addition; see config snippet above
    if bind_host is not None:
        if _is_loopback_host(bind_host):
            raise ConfigError(
                "grpc.bind_host resolves to a loopback address. "
                "For clustered deployments, set DISCO_GRPC__BIND_HOST to a routable address "
                "(e.g. Pod IP). For local dev, pass bind_host explicitly."
            )
        return bind_host

    inferred = _infer_non_loopback_ipv4()
    if inferred is None:
        raise ConfigError(
            "Unable to infer a non-loopback bind_host. "
            "Set DISCO_GRPC__BIND_HOST (recommended) or pass bind_host explicitly."
        )
    return inferred


def _pick_free_ports(bind_host: str, n: int) -> List[int]:
    """Ask the OS for `n` currently-free TCP ports on bind_host."""
    sockets: List[socket.socket] = []
    ports: List[int] = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind_host, 0))
            s.listen(1)
            sockets.append(s)
            ports.append(int(s.getsockname()[1]))
    finally:
        for s in sockets:
            try:
                s.close()
            except OSError:
                pass
    return ports


def _determine_worker_ports(
    *,
    bind_host: str,
    workers: Optional[int],
    ports: Optional[Sequence[int]],
) -> List[int]:
    if ports is not None:
        return [int(p) for p in ports]
    if workers is not None:
        count = int(workers)
    else:
        count = max(1, int(cpu_count()) - 1)
    return _pick_free_ports(bind_host, count)


def _build_worker_specs(bind_host: str, ports: Sequence[int]) -> List[WorkerSpec]:
    specs: List[WorkerSpec] = []
    for idx, port in enumerate(ports):
        addr = f"{bind_host}:{int(port)}"
        specs.append(WorkerSpec(address=addr, name=f"worker-{idx}"))
    return specs


def _worker_main(
    address: str,
    event_queues: Mapping[str, MPQueue],
    promise_queues: Mapping[str, MPQueue],
    settings: AppSettings,
    group: Optional[str],
    name: Optional[str],
    log_queue: Any,
    stop_event: Any,
) -> None:
    """Worker process entrypoint (must create its own Cluster client)."""
    mp_logging.configure_worker(log_queue)  # ✅ worker-side queue handler

    wlog = mp_logging.getLogger(__name__)
    wlog.info("Worker starting: %s", address)

    with Cluster.make_cluster(zookeeper_settings=settings.zookeeper, group=group) as cluster:
        worker = Worker(
            address=address,
            cluster=cluster,
            event_queues=event_queues,
            promise_queues=promise_queues,
            settings=settings,
            name=name,
        )
        worker.run_forever(stop_event)

    wlog.info("Worker exiting: %s", address)


def _orchestrator_process_entry(
        address: str,
        settings: AppSettings,
        group: Optional[str],
        stop_event: Any,
        log_queue: Any
) -> None:
    """Placeholder orchestrator: start, wait for stop_event, exit cleanly."""
    mp_logging.configure_worker(log_queue)
    olog = mp_logging.getLogger(__name__)
    olog.info("Orchestrator starting: %s", address)

    with Cluster.make_cluster(zookeeper_settings=settings.zookeeper, group=group) as cluster:
        orchestrator = Orchestrator(
            address=address,
            cluster=cluster,
            settings=settings.orchestrator,
        )
        orchestrator.run_forever(stop_event)

    olog.info("Orchestrator exiting: %s", address)


def _force_kill(proc: BaseProcess) -> None:
    """Best-effort hard kill of a process (platform-dependent)."""
    try:
        proc.kill()
        return
    except AttributeError:
        pass
    except Exception:
        pass

    pid = proc.pid
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


class Server:
    def __init__(
        self,
        settings: AppSettings,
        *,
        workers: Optional[int] = None,
        ports: Optional[List[int]] = None,
        bind_host: Optional[str] = None,
        group: Optional[str] = None,
        grace_s: Optional[int] = None,
        orchestrator: bool = True,
        loglevel: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._workers_arg = workers
        self._ports_arg = ports
        self._bind_host_arg = bind_host
        self._group = group or settings.zookeeper.default_group
        self._grace_s = float(grace_s) if grace_s is not None else float(settings.grace_s)
        self._start_orchestrator = orchestrator
        self._loglevel = loglevel

        self._shutdown_requested = threading.Event()
        self._cluster: Optional[Cluster] = None

        self._worker_specs: List[WorkerSpec] = []
        self._worker_procs: Dict[str, BaseProcess] = {}
        self._orchestrator_proc: Optional[BaseProcess] = None
        self._stop: Optional[Any] = None

    def start(self) -> None:
        self._acquire_singleton_guard()
        try:
            self._run()
        finally:
            self._release_singleton_guard()

    def _acquire_singleton_guard(self) -> None:
        global _SERVER_RUNNING
        with _SINGLETON_LOCK:
            if _SERVER_RUNNING:
                raise RuntimeError("A Server is already running in this process")
            _SERVER_RUNNING = True

    def _release_singleton_guard(self) -> None:
        global _SERVER_RUNNING
        with _SINGLETON_LOCK:
            _SERVER_RUNNING = False

    def _mp_context(self) -> Any:
        # Local spawn context: do not rely on global start method; avoids forking KazooClient.
        import multiprocessing as mp
        return mp.get_context("spawn")

    def _run(self) -> None:
        bind_host = _resolve_bind_host(self._settings, self._bind_host_arg)

        ports = _determine_worker_ports(
            bind_host=bind_host,
            workers=self._workers_arg,
            ports=self._ports_arg,
        )
        self._worker_specs = _build_worker_specs(bind_host, ports)

        ctx = self._mp_context()

        # Shared IPC queues: address -> Queue
        event_queues: Dict[str, MPQueue] = {}
        promise_queues: Dict[str, MPQueue] = {}
        for spec in self._worker_specs:
            event_queues[spec.address] = ctx.Queue()
            promise_queues[spec.address] = ctx.Queue()

        raw_level = self._loglevel if self._loglevel is not None else self._settings.logging.level
        log_level = _parse_log_level(raw_level)
        with mp_logging.setup_logging(level=log_level) as log_cfg:
            mp_logging.configure_worker(log_cfg.queue)
            logger.info("Server starting with %d workers (group=%s)", len(self._worker_specs), self._group)

            with Cluster.make_cluster(zookeeper_settings=self._settings.zookeeper, group=self._group) as cluster:
                self._cluster = cluster
                self._install_signal_handlers()

                stop_event = ctx.Event()
                self._stop = stop_event

                if self._start_orchestrator:
                    address = self._worker_specs[0].address
                    self._orchestrator_proc = ctx.Process(
                        name="orchestrator",
                        target=_orchestrator_process_entry,
                        args=(address, self._settings, self._group, stop_event, log_cfg.queue),
                        daemon=False,
                    )
                    self._orchestrator_proc.start()

                for spec in self._worker_specs:
                    proc = ctx.Process(
                        name=spec.name,
                        target=_worker_main,
                        args=(
                            spec.address,
                            event_queues,
                            promise_queues,
                            self._settings,
                            self._group,
                            spec.name,
                            log_cfg.queue,
                            stop_event,
                        ),
                        daemon=False,
                    )
                    proc.start()
                    self._worker_procs[spec.address] = proc

                try:
                    self._join_workers_forever()
                finally:
                    logger.info("Server stopping")

                    if self._stop is not None:
                        self._stop.set()
                    if self._orchestrator_proc is not None:
                        self._orchestrator_proc.join(timeout=5.0)
                    self._cluster = None

            logger.info("Server exited cleanly")

    def _install_signal_handlers(self) -> None:
        def handler(_signum: int, _frame: object) -> None:
            self._shutdown_requested.set()

        signal.signal(signal.SIGTERM, handler)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, handler)

    def _join_workers_forever(self) -> None:
        while True:
            if self._shutdown_requested.is_set():
                self._shutdown()
                return

            alive = False
            for proc in list(self._worker_procs.values()):
                proc.join(timeout=0.2)
                if proc.is_alive():
                    alive = True

            if not alive:
                return

    def _shutdown(self) -> None:
        # Signal all processes to stop gracefully via the shared stop event.
        if self._stop is not None:
            self._stop.set()

        deadline = time.monotonic() + self._grace_s

        # Cooperative exit window
        while time.monotonic() < deadline:
            if all(not p.is_alive() for p in self._worker_procs.values()):
                return
            time.sleep(0.2)

        # Escalation: terminate then kill
        for proc in self._worker_procs.values():
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

        kill_deadline = time.monotonic() + 2.0
        while time.monotonic() < kill_deadline:
            if all(not p.is_alive() for p in self._worker_procs.values()):
                return
            time.sleep(0.1)

        for proc in self._worker_procs.values():
            if proc.is_alive():
                _force_kill(proc)

        for proc in self._worker_procs.values():
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
