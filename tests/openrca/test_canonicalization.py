from vlm4rca.openrca.canonicalization import canonicalize_component, map_component_name


def test_canonicalize_component_normalizes_case_underscore_and_prefix() -> None:
    assert canonicalize_component("pod/Checkout_Service") == "checkout-service"
    assert canonicalize_component(" service:Tomcat01 ") == "tomcat01"


def test_map_component_exact_match_against_known_components() -> None:
    mapping = map_component_name("Tomcat01", ["Tomcat01", "Mysql02"])

    assert mapping.raw_ground_truth == "Tomcat01"
    assert mapping.mapped_component == "tomcat01"
    assert mapping.mapping_type == "component_exact"
    assert mapping.mapping_confidence == "exact"


def test_map_component_strips_resource_suffix_when_known_component_matches() -> None:
    mapping = map_component_name(
        "pod/payment-service-5f7c9d7b7c-abcde",
        ["payment-service", "checkout-service"],
    )

    assert mapping.mapped_component == "payment-service"
    assert mapping.mapping_type == "suffix_stripped"
    assert mapping.mapping_confidence == "heuristic"
