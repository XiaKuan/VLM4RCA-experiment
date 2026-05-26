from __future__ import annotations

from pathlib import Path

import pandas as pd

from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import (
    ShadowEdgeCandidate,
    SourceCandidate,
    TopologyExpansionConfig,
)
from vlm4rca.openrca.canonicalization import canonicalize_component


def _dedupe_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned = {
        (canonicalize_component(source), canonicalize_component(target))
        for source, target in edges
        if canonicalize_component(source) and canonicalize_component(target)
    }
    return sorted((source, target) for source, target in cleaned if source != target)


def topology_edges_from_shadow_edges(
    shadow_edges: list[ShadowEdgeCandidate],
) -> list[tuple[str, str]]:
    return _dedupe_edges([(edge.caller, edge.callee) for edge in shadow_edges])


def load_static_topology_edges(path: Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(path)
    lowered = {column.lower(): column for column in frame.columns}
    source_column = lowered.get("source") or lowered.get("caller") or lowered.get("from")
    target_column = lowered.get("target") or lowered.get("callee") or lowered.get("to")
    if source_column is None or target_column is None:
        raise ValueError("topology.csv must contain source/target columns")
    return _dedupe_edges(
        [(str(src), str(tgt)) for src, tgt in zip(frame[source_column], frame[target_column])]
    )


def _seed_reason(seed: RcaCandidate, direction: str) -> str:
    source = seed.introduced_by
    if source == "metric":
        suffix = "metric_hit"
    elif source == "trace":
        suffix = "trace_hit"
    elif source == "log":
        suffix = "log_hit"
    else:
        suffix = "candidate"
    return f"{direction}_of_{suffix}"


def _candidate_score(seed: RcaCandidate) -> float:
    if not seed.source_scores:
        return 0.1
    return round(max(float(score) for score in seed.source_scores.values()) * 0.5, 6)


def expand_topology_candidates(
    case_id: str,
    seed_candidates: list[RcaCandidate],
    edges: list[tuple[str, str]],
    config: TopologyExpansionConfig = TopologyExpansionConfig(),
) -> list[SourceCandidate]:
    if config.expand_hops != 1:
        raise ValueError("Phase 3 topology expansion supports exactly one hop")

    existing = {candidate.canonical_target for candidate in seed_candidates}
    emitted: dict[str, SourceCandidate] = {}
    normalized_edges = _dedupe_edges(edges)

    for seed in sorted(seed_candidates, key=lambda candidate: candidate.rank):
        if len(emitted) >= config.max_total_topology_candidates:
            break
        seed_target = seed.canonical_target
        neighbors: list[tuple[str, str, str]] = []
        upstream = sorted(source for source, target in normalized_edges if target == seed_target)
        downstream = sorted(target for source, target in normalized_edges if source == seed_target)
        neighbors.extend(
            (neighbor, "upstream", _seed_reason(seed, "upstream")) for neighbor in upstream
        )
        neighbors.extend(
            (neighbor, "downstream", _seed_reason(seed, "downstream")) for neighbor in downstream
        )

        per_seed_count = 0
        for neighbor, direction, reason in neighbors:
            if per_seed_count >= config.max_neighbors_per_candidate:
                break
            if len(emitted) >= config.max_total_topology_candidates:
                break
            if neighbor in existing or neighbor in emitted:
                continue
            emitted[neighbor] = SourceCandidate(
                case_id=case_id,
                candidate_key=f"service:{neighbor}",
                target_type="service",
                canonical_target=neighbor,
                raw_target=neighbor,
                source="topology",
                source_bucket="topology",
                score=_candidate_score(seed),
                evidence_summary=[f"{neighbor} is {direction} neighbor of {seed_target}"],
                source_refs=[f"topology:{direction}:{neighbor}:{seed_target}"],
                topology_reason=reason,
            )
            per_seed_count += 1

    return sorted(
        emitted.values(), key=lambda candidate: (-candidate.score, candidate.canonical_target)
    )
