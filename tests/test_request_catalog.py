from app.agents.request_types import (
    REQUEST_REGISTRY,
    RequestType,
    normalize_request_type,
)


def test_request_registry_has_measurements_and_triggers():
    measurements = set()
    for req_type, definition in REQUEST_REGISTRY.items():
        assert definition.title, f"{req_type} is missing a title"
        assert definition.description, f"{req_type} is missing a description"
        assert definition.measurement_id, f"{req_type} is missing measurement id"
        measurements.add(definition.measurement_id)
        if req_type != RequestType.CLARIFICATION:
            assert definition.triggers, f"{req_type} should define trigger phrases"
    assert len(measurements) == len(REQUEST_REGISTRY)


def test_normalize_request_type_aliases():
    assert normalize_request_type("price") == RequestType.PRICE_COMPARISON
    assert normalize_request_type("p&l") == RequestType.PNL
    assert normalize_request_type("details") == RequestType.ASSET_DETAILS
    assert normalize_request_type("general") == RequestType.GENERAL
    assert normalize_request_type("question", default=RequestType.CLARIFICATION) == RequestType.CLARIFICATION

