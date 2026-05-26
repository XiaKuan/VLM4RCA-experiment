from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlm4rca.openrca.canonicalization import map_component_name
from vlm4rca.openrca.models import GroundTruthMapping


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_raw_ground_truth(case_meta: Mapping[str, Any]) -> list[str]:
    matched_faults = case_meta.get("matched_faults") or []
    components = [
        str(fault["component"])
        for fault in matched_faults
        if isinstance(fault, Mapping) and fault.get("component")
    ]
    if components:
        return _dedupe_preserving_order(components)

    fallback = [str(component) for component in case_meta.get("ground_truth_components") or []]
    return _dedupe_preserving_order(fallback)


def build_ground_truth_mappings(case_meta: Mapping[str, Any]) -> list[GroundTruthMapping]:
    known_components = [
        str(component)
        for component in [
            *(case_meta.get("system_components") or []),
            *(case_meta.get("evidence_components") or []),
        ]
    ]
    return [
        map_component_name(raw_ground_truth, known_components)
        for raw_ground_truth in extract_raw_ground_truth(case_meta)
    ]
