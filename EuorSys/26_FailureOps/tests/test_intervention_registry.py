from failureops.intervention_registry import INTERVENTION_REGISTRY, get_intervention_spec
from failureops.runtime_service import P3_INTERVENTIONS


def test_registry_contains_all_p3_interventions():
    assert set(P3_INTERVENTIONS) == set(INTERVENTION_REGISTRY)


def test_runtime_policy_specs_preserve_event_record():
    for intervention in P3_INTERVENTIONS:
        spec = get_intervention_spec(intervention)
        if spec.intervention_class in {"runtime", "policy"}:
            assert spec.preserve_event_record
            assert "detector_events" in spec.required_invariants

