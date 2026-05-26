from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vlm4rca.candidates._column_utils import choose_column as _choose_column
from vlm4rca.candidates.multisource_models import SourceCandidate
from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import IncidentWindows

DEFAULT_LOG_KEYWORDS: tuple[str, ...] = (
    "timeout",
    "retry",
    "failed",
    "error",
    "exception",
    "unavailable",
    "deadline exceeded",
    "connection refused",
    "connection reset",
    "slow",
    "latency",
    "backoff",
    "circuit breaker",
)


@dataclass(frozen=True)
class LogBuildResult:
    case_id: str
    source_candidates: list[SourceCandidate]
    warnings: list[str]


def _duration_minutes(start: int, end: int) -> float:
    return max((end - start) / 60.0, 1.0)


def _keyword_counts(messages: pd.Series, keywords: tuple[str, ...]) -> dict[str, int]:
    lowered = messages.fillna("").astype(str).str.lower()
    counts: dict[str, int] = {}
    for keyword in keywords:
        count = int(lowered.str.contains(keyword, regex=False).sum())
        if count > 0:
            counts[keyword] = count
    return counts


def build_log_candidates_from_dataframe(
    case_id: str,
    logs: pd.DataFrame,
    windows: IncidentWindows,
    keywords: tuple[str, ...] = DEFAULT_LOG_KEYWORDS,
) -> LogBuildResult:
    timestamp_column = _choose_column(list(logs.columns), ("timestamp", "time", "ts"))
    service_column = _choose_column(
        list(logs.columns), ("service", "service_name", "component", "pod")
    )
    message_column = _choose_column(
        list(logs.columns), ("message", "msg", "body", "content", "log")
    )
    if timestamp_column is None or service_column is None or message_column is None:
        return LogBuildResult(
            case_id=case_id,
            source_candidates=[],
            warnings=["missing required log columns: timestamp, service, message"],
        )

    frame = logs.copy()
    frame[timestamp_column] = pd.to_numeric(frame[timestamp_column], errors="coerce")
    frame = frame.dropna(subset=[timestamp_column, service_column, message_column])
    baseline_mask = (frame[timestamp_column] >= windows.baseline_start) & (
        frame[timestamp_column] < windows.baseline_end
    )
    incident_mask = (frame[timestamp_column] >= windows.incident_start) & (
        frame[timestamp_column] < windows.incident_end
    )
    baseline_minutes = _duration_minutes(windows.baseline_start, windows.baseline_end)
    incident_minutes = _duration_minutes(windows.incident_start, windows.incident_end)

    candidates: list[SourceCandidate] = []
    for raw_service, group in frame.groupby(service_column, dropna=True):
        canonical = canonicalize_component(str(raw_service))
        if not canonical:
            continue
        baseline_counts = _keyword_counts(
            group.loc[baseline_mask.reindex(group.index, fill_value=False), message_column],
            keywords,
        )
        incident_counts = _keyword_counts(
            group.loc[incident_mask.reindex(group.index, fill_value=False), message_column],
            keywords,
        )
        deltas: dict[str, float] = {}
        for keyword in keywords:
            baseline_rate = baseline_counts.get(keyword, 0) / baseline_minutes
            incident_rate = incident_counts.get(keyword, 0) / incident_minutes
            delta = incident_rate - baseline_rate
            if delta > 0.0:
                deltas[keyword] = round(delta, 6)
        if not deltas:
            continue
        score = round(sum(deltas.values()), 6)
        top_keywords = sorted(deltas.items(), key=lambda item: (-item[1], item[0]))[:5]
        evidence = ", ".join(f"{keyword}={delta:.3f}/min" for keyword, delta in top_keywords)
        candidates.append(
            SourceCandidate(
                case_id=case_id,
                candidate_key=f"service:{canonical}",
                target_type="service",
                canonical_target=canonical,
                raw_target=str(raw_service),
                source="log",
                source_bucket="log",
                score=score,
                evidence_summary=[f"keyword_rate_delta {evidence}"],
                source_refs=[
                    f"log_keyword:{canonical}:{keyword}" for keyword, _delta in top_keywords
                ],
            )
        )

    return LogBuildResult(
        case_id=case_id,
        source_candidates=sorted(
            candidates, key=lambda candidate: (-candidate.score, candidate.canonical_target)
        ),
        warnings=[],
    )


def build_log_candidates_for_case(
    case_id: str,
    logs_path: Path,
    windows: IncidentWindows,
) -> LogBuildResult:
    logs = pd.read_csv(logs_path)
    return build_log_candidates_from_dataframe(case_id, logs, windows)
