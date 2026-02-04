from __future__ import annotations

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
from typing import Dict, List, Mapping, Optional, Sequence, Any

from disco.cluster import Cluster
from disco.config import AppSettings, ConfigError
from disco.worker import Worker, WorkerState


_SINGLETON_LOCK = threading.Lock()
_SERVER_RUNNING = False


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
            ip = s.getsockname()[0]
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
) -> None:
    """Worker process entrypoint (must create its own Cluster client)."""
    with Cluster.make_cluster(zookeeper_settings=settings.zookeeper, group=group) as cluster:
        worker = Worker(
            address=address,
            cluster=cluster,
            event_queues=event_queues,
            promise_queues=promise_queues,
            settings=settings,
            name=name,
        )
        worker.run_forever()


def _orchestrator_process_entry(stop_event: Any) -> None:
    """Placeholder orchestrator: start, wait for stop_event, exit cleanly."""
    stop_event.wait()


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
    ) -> None:
        self._settings = settings
        self._workers_arg = workers
        self._ports_arg = ports
        self._bind_host_arg = bind_host
        self._group = group
        self._grace_s = float(grace_s) if grace_s is not None else float(settings.grace_s)
        self._start_orchestrator = orchestrator

        self._shutdown_requested = threading.Event()
        self._cluster: Optional[Cluster] = None

        self._worker_specs: List[WorkerSpec] = []
        self._worker_procs: Dict[str, BaseProcess] = {}
        self._orchestrator_proc: Optional[BaseProcess] = None
        self._orchestrator_stop: Optional[Any] = None

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

        with Cluster.make_cluster(zookeeper_settings=self._settings.zookeeper, group=self._group) as cluster:
            self._cluster = cluster
            self._install_signal_handlers()

            if self._start_orchestrator:
                stop_event = ctx.Event()
                self._orchestrator_stop = stop_event
                self._orchestrator_proc = ctx.Process(
                    name="orchestrator",
                    target=_orchestrator_process_entry,
                    args=(stop_event,),
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
                    ),
                    daemon=False,
                )
                proc.start()
                self._worker_procs[spec.address] = proc

            try:
                self._join_workers_forever()
            finally:
                if self._orchestrator_stop is not None:
                    self._orchestrator_stop.set()
                if self._orchestrator_proc is not None:
                    self._orchestrator_proc.join(timeout=5.0)
                self._cluster = None

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
        cluster = self._cluster
        if cluster is not None:
            for addr in self._worker_procs.keys():
                try:
                    cluster.set_desired_state(worker_address=addr, state=WorkerState.TERMINATED)
                except Exception:
                    pass

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
