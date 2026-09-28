import sys

import pytest

import ray.train.health as health
from ray.train.health import (
    Action,
    ClusterProbe,
    Diagnose,
    Evict,
    HealthConfig,
    HealthPolicy,
    HealthState,
    NodeHealth,
    NodeProbe,
    Noop,
    OnDemandProbe,
    Probe,
    ProbeResult,
    Reattempt,
    WorkerHealth,
    WorkerProbe,
)
from ray.train.health.decision import SEVERITY


class HostTemp(NodeProbe):
    def poll(self, ctx):
        return None


class Queues(ClusterProbe):
    name = "Queues"

    def poll(self, ctx):
        return {}


class Loss(WorkerProbe):
    def poll(self):
        return None


class Screen(OnDemandProbe):
    def poll(self, ctx):
        return ProbeResult()


def test_probe_name_defaults_to_the_class_name():
    assert HostTemp.probe_name() == "HostTemp"
    assert Queues.probe_name() == "Queues"


def test_on_demand_probe_defaults():
    probe = Screen()
    assert probe.scope == health.NODE_SCOPE
    assert probe.stop_workers is False


def test_decisions_carry_their_action():
    assert Noop().action is Action.NOOP
    assert Reattempt().action is Action.REATTEMPT
    assert Evict(target_nodes=["n1"]).action is Action.EVICT
    assert Diagnose(on_demand_probes=[Screen()]).action is Action.DIAGNOSE


def test_severity_orders_actions():
    order = sorted(Action, key=SEVERITY.get)
    assert order == [Action.NOOP, Action.DIAGNOSE, Action.REATTEMPT, Action.EVICT]


@pytest.mark.parametrize("probe", [HostTemp, HostTemp()])
def test_results_accepts_a_probe_class_or_instance(probe):
    state = HealthState(
        nodes={"nB": NodeHealth("nB", 1.0, {"HostTemp": ProbeResult(detail="x")})}
    )
    assert state.results(probe)["nB"].detail == "x"


def test_results_rejects_a_probe_of_no_known_kind():
    class Unknown(Probe):
        pass

    with pytest.raises(TypeError, match="Unknown"):
        HealthState().results(Unknown)


def test_health_state_typed_reads():
    state = HealthState(
        workers={
            0: WorkerHealth(
                0,
                "nA",
                1.0,
                step=10,
                probe_results={"Loss": ProbeResult(detail="w")},
                reported={"grad_norm": 1.8},
            ),
            1: WorkerHealth(1, "nB", 1.0, step=10),
        },
        nodes={"nB": NodeHealth("nB", 1.0, {"HostTemp": ProbeResult(detail="x")})},
        cluster={"Queues": {"q1": ProbeResult(detail="y")}},
        on_demand_probes={"Screen": {"nA": ProbeResult(passed=False)}},
    )
    assert state.results(Loss) == {0: ProbeResult(detail="w")}
    assert state.results(HostTemp)["nB"].detail == "x"
    assert state.results(Queues)["q1"].detail == "y"
    assert state.results(Screen)["nA"].passed is False
    assert state.reported == {0: {"grad_norm": 1.8}}
    assert state.ranks_on("nB") == [1]
    assert state.node_of(0) == "nA"
    assert state.node_of(7) is None


def test_health_config_holds_policies():
    policy = HealthPolicy(probe_creator=lambda: [Screen()], preflight=True)
    assert HealthConfig(policies=[policy]).policies == [policy]
    assert HealthConfig().policies == []


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
