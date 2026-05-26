from __future__ import annotations

import re

from vlm4rca.openrca.models import GroundTruthMapping

_PREFIXES = ("pod/", "container/", "service/", "service:", "resource:")
_HASH_SEGMENT_RE = re.compile(r"^[a-f0-9]{5,}$")
_ORDINAL_RE = re.compile(r"^\d+$")


def canonicalize_component(raw: str) -> str:
    value = raw.strip()
    lowered = value.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.lower().strip("-")


def strip_resource_suffix(canonical_name: str) -> str:
    parts = canonical_name.split("-")
    if (
        len(parts) >= 3
        and _HASH_SEGMENT_RE.fullmatch(parts[-2])
        and _HASH_SEGMENT_RE.fullmatch(parts[-1])
    ):
        return "-".join(parts[:-2])
    if len(parts) >= 2 and _ORDINAL_RE.fullmatch(parts[-1]):
        return "-".join(parts[:-1])
    return canonical_name


def map_component_name(raw_ground_truth: str, known_components: list[str]) -> GroundTruthMapping:
    canonical_known = {
        canonicalize_component(component): component for component in known_components
    }
    canonical_raw = canonicalize_component(raw_ground_truth)

    if canonical_raw in canonical_known:
        return GroundTruthMapping(
            raw_ground_truth=raw_ground_truth,
            mapped_component=canonical_raw,
            mapping_type="component_exact",
            mapping_confidence="exact",
            source="case_meta.matched_faults.component",
        )

    stripped = strip_resource_suffix(canonical_raw)
    if stripped in canonical_known:
        return GroundTruthMapping(
            raw_ground_truth=raw_ground_truth,
            mapped_component=stripped,
            mapping_type="suffix_stripped",
            mapping_confidence="heuristic",
            source="case_meta.matched_faults.component",
        )

    return GroundTruthMapping(
        raw_ground_truth=raw_ground_truth,
        mapped_component=canonical_raw,
        mapping_type="unknown",
        mapping_confidence="unknown",
        source="case_meta.matched_faults.component",
    )
