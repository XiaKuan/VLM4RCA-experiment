from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import ShadowEdgeCandidate
from vlm4rca.evaluation.recall import (
    candidates_by_case_to_targets,
    evaluate_component_recall_at_k,
)
from vlm4rca.openrca.models import Phase1CaseSidecar

VARIANT_CHAIN = ("M", "M+T", "M+T+L", "M+T+L+Topo")


def _eligible_for_variant(sidecar: Phase1CaseSidecar, variant: str) -> bool:
    availability = sidecar.modality_availability
    if variant == "M":
        return availability.metrics_available
    if variant == "M+T":
        return availability.metrics_available and availability.traces_available
    if variant == "M+T+L":
        return (
            availability.metrics_available
            and availability.traces_available
            and availability.logs_available
        )
    if variant == "M+T+L+Topo":
        return (
            availability.metrics_available
            and availability.traces_available
            and availability.logs_available
            and availability.topology_available
        )
    raise ValueError(f"Unknown variant: {variant}")


def _recall_row(
    variant: str,
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_case: Mapping[str, Sequence[RcaCandidate]],
) -> dict[str, Any]:
    summary = evaluate_component_recall_at_k(
        sidecars,
        candidates_by_case_to_targets(candidates_by_case),
        ks=[3, 5, 8],
    )
    soft_sidecars = [
        sidecar for sidecar in sidecars if sidecar.case_group == "soft_latency"
    ]
    soft_summary = evaluate_component_recall_at_k(
        soft_sidecars,
        candidates_by_case_to_targets(candidates_by_case),
        ks=[8],
    )
    denominator = len(sidecars)
    total_candidates = sum(
        len(candidates_by_case.get(sidecar.case_id, [])) for sidecar in sidecars
    )
    return {
        "variant": variant,
        "n_cases": denominator,
        "recall_at_3": summary.component_recall_at_k["3"],
        "recall_at_5": summary.component_recall_at_k["5"],
        "recall_at_8": summary.component_recall_at_k["8"],
        "soft_recall_at_8": soft_summary.component_recall_at_k["8"],
        "avg_candidates": (
            round(total_candidates / denominator, 6) if denominator else 0.0
        ),
    }


def build_candidate_recall_ablation(
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_variant: Mapping[str, Mapping[str, Sequence[RcaCandidate]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    primary: dict[str, dict[str, Any]] = {}
    secondary: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_CHAIN:
        candidates = candidates_by_variant.get(variant, {})
        primary_sidecars = [
            sidecar
            for sidecar in sidecars
            if _eligible_for_variant(sidecar, variant)
        ]
        primary[variant] = _recall_row(variant, primary_sidecars, candidates)
        secondary[variant] = _recall_row(variant, sidecars, candidates)
    return {"primary": primary, "secondary": secondary}


def _case_hit(
    sidecar: Phase1CaseSidecar,
    candidates: Sequence[RcaCandidate],
    k: int = 8,
) -> bool:
    targets = {mapping.mapped_component for mapping in sidecar.gt_mapping}
    top_k = {
        candidate.canonical_target
        for candidate in sorted(candidates, key=lambda item: item.rank)[:k]
    }
    return bool(targets.intersection(top_k))


def build_first_hit_rows(
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_variant: Mapping[str, Mapping[str, Sequence[RcaCandidate]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sidecar in sidecars:
        hits = {
            variant: _case_hit(
                sidecar,
                candidates_by_variant.get(variant, {}).get(sidecar.case_id, []),
                k=8,
            )
            for variant in VARIANT_CHAIN
        }
        if hits["M"]:
            first_hit_source = "Metric"
            new_hit_source = "None"
        elif hits["M+T"]:
            first_hit_source = "Trace"
            new_hit_source = "Trace"
        elif hits["M+T+L"]:
            first_hit_source = "Log"
            new_hit_source = "Log"
        elif hits["M+T+L+Topo"]:
            first_hit_source = "Topology"
            new_hit_source = "Topology"
        else:
            first_hit_source = "NotHit"
            new_hit_source = "NotHit"
        rows.append(
            {
                "case_id": sidecar.case_id,
                "case_group": sidecar.case_group,
                "ground_truth_component": ",".join(
                    mapping.mapped_component for mapping in sidecar.gt_mapping
                ),
                "metric_hit_at_8": hits["M"],
                "metric_trace_hit_at_8": hits["M+T"],
                "metric_trace_log_hit_at_8": hits["M+T+L"],
                "full_hit_at_8": hits["M+T+L+Topo"],
                "first_hit_source": first_hit_source,
                "new_hit_source": new_hit_source,
            }
        )
    return rows


def build_checkpoint_decision(
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_variant: Mapping[str, Mapping[str, Sequence[RcaCandidate]]],
) -> dict[str, Any]:
    eligible_sidecars = [
        sidecar
        for sidecar in sidecars
        if _eligible_for_variant(sidecar, "M+T+L+Topo")
    ]
    metric_row = _recall_row(
        "M", eligible_sidecars, candidates_by_variant.get("M", {})
    )
    full_row = _recall_row(
        "M+T+L+Topo",
        eligible_sidecars,
        candidates_by_variant.get("M+T+L+Topo", {}),
    )
    proceed = full_row["recall_at_8"] > metric_row["recall_at_8"]
    blockers = []
    if not proceed:
        blockers.append(
            "Full Recall@8 did not improve over Metric-only on Full-eligible pilot cases; "
            "pause VLM and plotting and inspect trace parser, canonicalization, log keywords, "
            "and topology expansion."
        )
    return {
        "eligible_cases": len(eligible_sidecars),
        "metric_recall_at_8": metric_row["recall_at_8"],
        "full_recall_at_8": full_row["recall_at_8"],
        "proceed_to_vlm_and_rendering": proceed,
        "blockers": blockers,
    }


def summarize_shadow_edges(
    relevant_edges_by_case: Mapping[str, Sequence[str]],
    shadow_edges_by_case: Mapping[str, Sequence[ShadowEdgeCandidate]],
) -> dict[str, Any]:
    ks = (3, 5, 8)
    hit_counts = {k: 0 for k in ks}
    total_cases = len(relevant_edges_by_case)
    for case_id, relevant_edges in relevant_edges_by_case.items():
        relevant = set(relevant_edges)
        ranked_edges = sorted(
            shadow_edges_by_case.get(case_id, []), key=lambda edge: edge.rank
        )
        for k in ks:
            top_k = {edge.edge_key for edge in ranked_edges[:k]}
            if relevant.intersection(top_k):
                hit_counts[k] += 1
    return {
        "diagnostic_only": True,
        "n_cases": total_cases,
        "edge_recall_at_3": hit_counts[3] / total_cases if total_cases else 0.0,
        "edge_recall_at_5": hit_counts[5] / total_cases if total_cases else 0.0,
        "edge_recall_at_8": hit_counts[8] / total_cases if total_cases else 0.0,
    }
