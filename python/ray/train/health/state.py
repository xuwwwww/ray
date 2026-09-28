from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from ray.train.health.probe import (
    ClusterProbe,
    NodeProbe,
    OnDemandProbe,
    Probe,
    ProbeResult,
    WorkerProbe,
)
from ray.util.annotations import DeveloperAPI, PublicAPI


@DeveloperAPI
@dataclass
class WorkerHealth:
    """One worker's snapshot, carried on ``WorkerStatus.health``."""

    worker_rank: int
    node_id: str
    snapshot_at: float
    step: Optional[int] = None
    probe_results: Dict[str, ProbeResult] = field(default_factory=dict)
    reported: Dict[str, Any] = field(default_factory=dict)


@DeveloperAPI
@dataclass
class NodeHealth:
    node_id: str
    snapshot_at: float
    probe_results: Dict[str, ProbeResult] = field(default_factory=dict)


@PublicAPI(stability="alpha")
@dataclass
class HealthState:
    """One poll's worth of evidence: the latest snapshot per source.

    Attributes:
        workers: ``{world_rank: WorkerHealth}``.
        nodes: ``{node_id: NodeHealth}``.
        cluster: ``{probe_name: {entity_id: ProbeResult}}`` from cluster probes.
        on_demand_probes: ``{probe_name: {entity_id: ProbeResult}}``.
    """

    workers: Mapping[int, WorkerHealth] = field(default_factory=dict)
    nodes: Mapping[str, NodeHealth] = field(default_factory=dict)
    cluster: Mapping[str, Dict[str, ProbeResult]] = field(default_factory=dict)
    on_demand_probes: Mapping[str, Dict[str, ProbeResult]] = field(default_factory=dict)

    def results(self, probe: Union[Probe, Type[Probe]]) -> Dict[Any, ProbeResult]:
        """Latest results of ``probe``, found by its kind.

        Args:
            probe: The probe class, or an instance of it.

        Returns:
            Results keyed by ``world_rank`` for a ``WorkerProbe``, by
            ``node_id`` for a ``NodeProbe``, and by entity for a
            ``ClusterProbe`` or an ``OnDemandProbe``.

        Raises:
            TypeError: If ``probe`` is none of those four kinds.
        """
        kind = probe if isinstance(probe, type) else type(probe)
        name = kind.probe_name()
        if issubclass(kind, ClusterProbe):
            return dict(self.cluster.get(name, {}))
        if issubclass(kind, OnDemandProbe):
            return dict(self.on_demand_probes.get(name, {}))
        if issubclass(kind, WorkerProbe):
            sources = self.workers
        elif issubclass(kind, NodeProbe):
            sources = self.nodes
        else:
            raise TypeError(
                f"{kind.__name__} is not a WorkerProbe, NodeProbe, ClusterProbe "
                "or OnDemandProbe."
            )
        return {
            key: source.probe_results[name]
            for key, source in sources.items()
            if name in source.probe_results
        }

    @property
    def reported(self) -> Dict[int, Dict[str, Any]]:
        """``{world_rank: metrics}`` from ``ray.train.health.report()``."""
        return {
            rank: worker.reported
            for rank, worker in self.workers.items()
            if worker.reported
        }

    def ranks_on(self, node_id: str) -> List[int]:
        """The world ranks on a node.

        Args:
            node_id: The node to look up.

        Returns:
            The ranks on ``node_id``, sorted; empty if none.
        """
        return sorted(r for r, w in self.workers.items() if w.node_id == node_id)

    def node_of(self, rank: int) -> Optional[str]:
        """The node a rank runs on.

        Args:
            rank: The world rank to look up.

        Returns:
            The node id, or ``None`` if the rank has not reported.
        """
        worker = self.workers.get(rank)
        return worker.node_id if worker else None
