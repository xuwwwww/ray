import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ray.util.annotations import DeveloperAPI, PublicAPI

# ----------------------------------------------------------------------
# Base
# ----------------------------------------------------------------------


@PublicAPI(stability="alpha")
@dataclass
class ProbeResult:
    """What every probe returns.

    Attributes:
        metrics: Scalar signals for the whole entity.
        devices: Per-device signals, keyed by device index or UUID.
        events: Discrete, named occurrences since the last sample.
        passed: Pass/fail verdict. ``None`` means the probe takes no position.
        detail: Free text for humans.
        artifacts: Paths to files the probe wrote, such as stack dumps.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    devices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    passed: Optional[bool] = None
    detail: str = ""
    artifacts: List[str] = field(default_factory=list)


@PublicAPI(stability="alpha")
class Probe(abc.ABC):
    """Base class for every collector.

    ``name``, set on the class, is the key results are filed under. It defaults
    to the class name, so instances of one probe class share a key.
    """

    name: str = ""

    @classmethod
    def probe_name(cls) -> str:
        return cls.name or cls.__name__


# ----------------------------------------------------------------------
# WorkerProbe
# ----------------------------------------------------------------------


@PublicAPI(stability="alpha")
class WorkerProbe(Probe):
    """Runs in each train worker, on every poll."""

    @abc.abstractmethod
    def poll(self) -> Optional[ProbeResult]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# NodeProbe
# ----------------------------------------------------------------------


@DeveloperAPI
@dataclass(frozen=True)
class NodeContext:
    node_id: str


@PublicAPI(stability="alpha")
class NodeProbe(Probe):
    """Runs in each node's ``NodeMonitor``, outside the worker process."""

    interval_s: float = 10.0

    @abc.abstractmethod
    def poll(self, ctx: NodeContext) -> Optional[ProbeResult]:
        """Sample this node.

        Args:
            ctx: The node being sampled.

        Returns:
            The sample, or ``None`` if there is nothing to report.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# ClusterProbe
# ----------------------------------------------------------------------


@DeveloperAPI
@dataclass(frozen=True)
class ClusterContext:
    """The run a cluster probe is watching.

    Attributes:
        node_ids: Nodes hosting at least one rank of the worker group.
        rank_to_node: ``{world_rank: node_id}`` for the worker group.
    """

    node_ids: List[str] = field(default_factory=list)
    rank_to_node: Dict[int, str] = field(default_factory=dict)


@PublicAPI(stability="alpha")
class ClusterProbe(Probe):
    """Runs on the controller and reports many entities in one read.

    Returns ``{entity_id: ProbeResult}``, keyed by whatever the probe measures
    (a node, a communicator, a queue). Results land in ``HealthState.cluster``.
    Polled on a background thread, so it may keep state between polls.
    """

    interval_s: float = 10.0

    @abc.abstractmethod
    def poll(self, ctx: ClusterContext) -> Dict[str, ProbeResult]:
        """Read the source once for the whole run.

        Args:
            ctx: The ranks and nodes of the current worker group.

        Returns:
            ``{entity_id: ProbeResult}`` for every entity the source reports.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# OnDemandProbe
# ----------------------------------------------------------------------

NODE_SCOPE = "NODE"
WORKER_SCOPE = "WORKER"


@DeveloperAPI
@dataclass(frozen=True)
class OnDemandProbeContext:
    """Passed to an on-demand probe when it is pushed.

    Attributes:
        entity_id: The node id for a node-scoped probe, the rank otherwise.
        node_id: The node the probe runs on.
        rank: The world rank, for a worker-scoped probe.
        nodes: ``{node_id: [world_rank, ...]}`` for every node in this push.
        timeout_s: The budget for this invocation.
    """

    entity_id: str = ""
    node_id: str = ""
    rank: Optional[int] = None
    nodes: Dict[str, List[int]] = field(default_factory=dict)
    timeout_s: float = 60.0


@PublicAPI(stability="alpha")
class OnDemandProbe(Probe):
    """An active check the controller pushes: a diagnostic or pre-flight check.

    Attributes:
        stop_workers: The check needs the accelerator, so workers are paused.
        timeout_s: Hard timeout.
        scope: ``NODE_SCOPE`` runs once per node; ``WORKER_SCOPE`` runs inside
            each targeted training worker, for checks that must attach to it.
    """

    stop_workers: bool = False
    timeout_s: float = 60.0
    scope: str = NODE_SCOPE

    @abc.abstractmethod
    def poll(self, ctx: OnDemandProbeContext) -> ProbeResult:
        """Run the check once, where it was pushed.

        Args:
            ctx: Where the check runs and what it is about.

        Returns:
            The result of the check.
        """
        raise NotImplementedError
