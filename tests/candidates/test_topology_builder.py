from pathlib import Path

import pandas as pd

from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import ShadowEdgeCandidate, TopologyExpansionConfig
from vlm4rca.candidates.topology_builder import (
    expand_topology_candidates,
    load_static_topology_edges,
    topology_edges_from_shadow_edges,
)


def _candidate(rank: int, target: str, introduced_by: str = "metric") -> RcaCandidate:
    return RcaCandidate(
        case_id="case_001",
        variant="M+T+L",
        candidate_key=f"service:{target}",
        variant_candidate_id=f"cand:case_001:M+T+L:{rank}:service:{target}",
        target_type="service",
        canonical_target=target,
        raw_target=target,
        introduced_by=introduced_by,
        metric_only_present=introduced_by == "metric",
        present_in_variants=["M+T+L"],
        sources=[introduced_by],
        source_scores={introduced_by: 1.0},
        evidence_summary=[f"{introduced_by} evidence"],
        rank=rank,
        selected_for_rendering=True,
    )


def _edge(rank: int, caller: str, callee: str, score: float = 1.0) -> ShadowEdgeCandidate:
    return ShadowEdgeCandidate(
        case_id="case_001",
        edge_key=f"edge:{caller}->{callee}",
        caller=caller,
        callee=callee,
        raw_caller=caller,
        raw_callee=callee,
        source_score=score,
        evidence_summary=["edge evidence"],
        rank=rank,
    )


def test_topology_edges_from_shadow_edges_are_deduped_and_sorted() -> None:
    edges = topology_edges_from_shadow_edges(
        [
            _edge(2, "gateway", "checkout"),
            _edge(1, "gateway", "checkout"),
            _edge(3, "checkout", "payment"),
        ]
    )

    assert edges == [("checkout", "payment"), ("gateway", "checkout")]


def test_load_static_topology_edges_reads_source_target_csv(tmp_path: Path) -> None:
    path = tmp_path / "topology.csv"
    pd.DataFrame({"source": ["Gateway", "Checkout"], "target": ["Checkout", "Payment"]}).to_csv(
        path, index=False
    )

    edges = load_static_topology_edges(path)

    assert edges == [("checkout", "payment"), ("gateway", "checkout")]


def test_expand_topology_candidates_adds_one_hop_neighbors_with_reasons() -> None:
    seeds = [_candidate(1, "checkout"), _candidate(2, "redis")]
    edges = [("gateway", "checkout"), ("checkout", "payment")]

    candidates = expand_topology_candidates(
        "case_001",
        seeds,
        edges,
        config=TopologyExpansionConfig(
            max_neighbors_per_candidate=2, max_total_topology_candidates=5
        ),
    )

    assert [candidate.canonical_target for candidate in candidates] == ["gateway", "payment"]
    assert candidates[0].source == "topology"
    assert candidates[0].source_bucket == "topology"
    assert candidates[0].topology_reason == "upstream_of_metric_hit"
    assert candidates[1].topology_reason == "downstream_of_metric_hit"


def test_expand_topology_candidates_respects_global_cap_and_skips_existing_seeds() -> None:
    seeds = [_candidate(1, "checkout")]
    edges = [
        ("a", "checkout"),
        ("b", "checkout"),
        ("checkout", "c"),
        ("checkout", "d"),
        ("checkout", "checkout"),
    ]

    candidates = expand_topology_candidates(
        "case_001",
        seeds,
        edges,
        config=TopologyExpansionConfig(
            max_neighbors_per_candidate=4, max_total_topology_candidates=3
        ),
    )

    assert len(candidates) == 3
    assert "checkout" not in [candidate.canonical_target for candidate in candidates]
