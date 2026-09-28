import abc
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ray.train.health.decision import HealthDecision
from ray.train.health.probe import Probe
from ray.train.health.state import HealthState
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class Evaluator(abc.ABC):
    """Judges a ``HealthState`` and returns zero or more decisions.

    An evaluator lives for the whole run, so it may keep its own history
    across polls. If ``evaluate()`` raises, the error is logged and the
    evaluator is not called again for the rest of the run; training continues.
    """

    @abc.abstractmethod
    def evaluate(self, state: HealthState) -> List[HealthDecision]:
        """Judge the run's current health.

        Args:
            state: The latest evidence from every probe and the training loop.

        Returns:
            Zero or more decisions; an empty list when nothing is wrong.
        """
        raise NotImplementedError

    def on_worker_group_start(self) -> None:
        """Clear history kept from the previous worker group. Called on every
        (re)start."""


ProbeCreator = Callable[[], List[Probe]]
EvaluatorCreator = Callable[[], List[Evaluator]]


@PublicAPI(stability="alpha")
@dataclass
class HealthPolicy:
    """A bundle of probes and the evaluators that judge them.

    Attributes:
        probe_creator: Builds the policy's probes, once per run.
        evaluator_creator: Builds the policy's evaluators, once per run.
        preflight: Also run the policy's ``OnDemandProbe``s on every candidate
            node before its first worker group is scheduled there.
    """

    probe_creator: Optional[ProbeCreator] = None
    evaluator_creator: Optional[EvaluatorCreator] = None
    preflight: bool = False


@PublicAPI(stability="alpha")
@dataclass
class HealthConfig:
    """The health policies for a run. Empty means off."""

    policies: List[HealthPolicy] = field(default_factory=list)
