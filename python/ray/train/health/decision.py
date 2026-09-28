from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, List

from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.train.health.probe import OnDemandProbe


@PublicAPI(stability="alpha")
class Action(Enum):
    NOOP = auto()
    DIAGNOSE = auto()
    REATTEMPT = auto()
    EVICT = auto()


SEVERITY = {Action.NOOP: 0, Action.DIAGNOSE: 1, Action.REATTEMPT: 2, Action.EVICT: 3}


@PublicAPI(stability="alpha")
@dataclass
class HealthDecision:
    action: ClassVar[Action]
    reason: str = ""


@PublicAPI(stability="alpha")
@dataclass
class Noop(HealthDecision):
    action: ClassVar[Action] = Action.NOOP


@PublicAPI(stability="alpha")
@dataclass
class Reattempt(HealthDecision):
    action: ClassVar[Action] = Action.REATTEMPT


@PublicAPI(stability="alpha")
@dataclass
class Evict(HealthDecision):
    action: ClassVar[Action] = Action.EVICT
    target_nodes: List[str] = field(default_factory=list)


@PublicAPI(stability="alpha")
@dataclass
class Diagnose(HealthDecision):
    action: ClassVar[Action] = Action.DIAGNOSE
    on_demand_probes: List["OnDemandProbe"] = field(default_factory=list)
    target_nodes: List[str] = field(default_factory=list)
    target_ranks: List[int] = field(default_factory=list)
