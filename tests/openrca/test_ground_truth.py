from vlm4rca.openrca.ground_truth import build_ground_truth_mappings, extract_raw_ground_truth


def test_extracts_unique_raw_ground_truth_from_matched_faults() -> None:
    case_meta = {
        "matched_faults": [
            {"component": "apache02", "reason": "network packet loss", "timestamp": 1},
            {"component": "apache02", "reason": "network latency", "timestamp": 2},
            {"component": "MG01", "reason": "network latency", "timestamp": 3},
        ]
    }

    assert extract_raw_ground_truth(case_meta) == ["apache02", "MG01"]


def test_builds_ground_truth_mapping_sidecar_entries() -> None:
    case_meta = {
        "matched_faults": [{"component": "Tomcat01", "reason": "network latency", "timestamp": 1}],
        "system_components": ["Tomcat01", "Mysql02"],
        "evidence_components": ["Tomcat01", "Redis01"],
    }

    mappings = build_ground_truth_mappings(case_meta)

    assert len(mappings) == 1
    assert mappings[0].raw_ground_truth == "Tomcat01"
    assert mappings[0].mapped_component == "tomcat01"
    assert mappings[0].source == "case_meta.matched_faults.component"
