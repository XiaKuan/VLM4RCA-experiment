from __future__ import annotations

from collections.abc import Mapping, Sequence

from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import Phase1CaseSidecar, RecallAtKSummary, RecallCaseResult


def evaluate_component_recall_at_k(
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_case: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (3, 5, 8),
) -> RecallAtKSummary:
    ordered_ks = sorted(int(k) for k in ks)
    case_results: list[RecallCaseResult] = []
    hit_counts = {str(k): 0 for k in ordered_ks}

    for sidecar in sidecars:
        targets = [mapping.mapped_component for mapping in sidecar.gt_mapping]
        candidates = [
            canonicalize_component(candidate)
            for candidate in candidates_by_case.get(sidecar.case_id, [])
        ]
        hit_at_k: dict[str, bool] = {}
        hit_targets_for_case: set[str] = set()

        for k in ordered_ks:
            top_k = set(candidates[:k])
            hits = [target for target in targets if target in top_k]
            hit = bool(hits)
            hit_at_k[str(k)] = hit
            if hit:
                hit_counts[str(k)] += 1
                hit_targets_for_case.update(hits)

        missed_targets = [target for target in targets if target not in hit_targets_for_case]
        case_results.append(
            RecallCaseResult(
                case_id=sidecar.case_id,
                hit_at_k=hit_at_k,
                hit_targets=sorted(hit_targets_for_case),
                missed_targets=missed_targets,
            )
        )

    denominator = len(sidecars)
    recall = {
        str(k): (hit_counts[str(k)] / denominator if denominator else 0.0) for k in ordered_ks
    }
    return RecallAtKSummary(
        n_cases=denominator,
        ks=ordered_ks,
        component_recall_at_k=recall,
        case_results=case_results,
    )
