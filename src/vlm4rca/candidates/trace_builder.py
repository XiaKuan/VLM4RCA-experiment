from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vlm4rca.candidates.multisource_models import ShadowEdgeCandidate, SourceCandidate
from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import IncidentWindows

EPSILON = 1e-9
MIN_POINTS_PER_WINDOW = 2


@dataclass(frozen=True)
class TraceBuildResult:
    case_id: str
    source_candidates: list[SourceCandidate]
    shadow_edges: list[ShadowEdgeCandidate]
    warnings: list[str]


@dataclass(frozen=True)
class TraceColumns:
    timestamp: str
    service: str
    duration: str
    trace_id: str | None
    span_id: str | None
    parent_span_id: str | None
    caller: str | None
    callee: str | None


def _choose_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _detect_columns(frame: pd.DataFrame) -> TraceColumns | None:
    columns = list(frame.columns)
    timestamp = _choose_column(columns, ("timestamp", "time", "start_time", "starttime"))
    service = _choose_column(columns, ("service", "service_name", "servicename", "process_serviceName"))
    duration = _choose_column(columns, ("duration", "duration_ms", "elapsed", "span_duration"))
    if timestamp is None or service is None or duration is None:
        return None
    return TraceColumns(
        timestamp=timestamp,
        service=service,
        duration=duration,
        trace_id=_choose_column(columns, ("trace_id", "traceid")),
        span_id=_choose_column(columns, ("span_id", "spanid")),
        parent_span_id=_choose_column(columns, ("parent_span_id", "parentspanid", "parent_id")),
        caller=_choose_column(columns, ("caller", "caller_service", "parent_service")),
        callee=_choose_column(columns, ("callee", "callee_service", "child_service")),
    )


def _finite_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _relative_shift(baseline_value: float, incident_value: float) -> float:
    denominator = abs(baseline_value)
    if denominator <= EPSILON:
        denominator = 1.0
    return max(0.0, (incident_value - baseline_value) / denominator)


def _score_duration_shift(
    baseline: pd.Series, incident: pd.Series
) -> tuple[float, float, float, int, int]:
    baseline_values = _finite_values(baseline)
    incident_values = _finite_values(incident)
    if (
        baseline_values.size < MIN_POINTS_PER_WINDOW
        or incident_values.size < MIN_POINTS_PER_WINDOW
    ):
        return (
            0.0,
            0.0,
            0.0,
            int(baseline_values.size),
            int(incident_values.size),
        )
    baseline_p95 = _p95(baseline_values)
    incident_p95 = _p95(incident_values)
    relative = _relative_shift(baseline_p95, incident_p95)
    score = round(
        relative + max(0.0, incident_p95 - baseline_p95) / max(abs(baseline_p95), 1.0),
        6,
    )
    return (
        score,
        round(baseline_p95, 6),
        round(incident_p95, 6),
        int(baseline_values.size),
        int(incident_values.size),
    )


def _window_masks(
    frame: pd.DataFrame, timestamp_column: str, windows: IncidentWindows
) -> tuple[pd.Series, pd.Series]:
    timestamps = pd.to_numeric(frame[timestamp_column], errors="coerce")
    baseline_mask = (timestamps >= windows.baseline_start) & (
        timestamps < windows.baseline_end
    )
    incident_mask = (timestamps >= windows.incident_start) & (
        timestamps < windows.incident_end
    )
    return baseline_mask, incident_mask


def _build_service_candidates(
    case_id: str,
    frame: pd.DataFrame,
    columns: TraceColumns,
    windows: IncidentWindows,
) -> list[SourceCandidate]:
    baseline_mask, incident_mask = _window_masks(frame, columns.timestamp, windows)
    candidates: list[SourceCandidate] = []
    for raw_service, group in frame.groupby(columns.service, dropna=True):
        canonical = canonicalize_component(str(raw_service))
        if not canonical:
            continue
        baseline = group.loc[
            baseline_mask.reindex(group.index, fill_value=False), columns.duration
        ]
        incident = group.loc[
            incident_mask.reindex(group.index, fill_value=False), columns.duration
        ]
        score, baseline_p95, incident_p95, baseline_points, incident_points = (
            _score_duration_shift(baseline, incident)
        )
        if score <= 0.0:
            continue
        candidates.append(
            SourceCandidate(
                case_id=case_id,
                candidate_key=f"service:{canonical}",
                target_type="service",
                canonical_target=canonical,
                raw_target=str(raw_service),
                source="trace",
                source_bucket="trace_service",
                score=score,
                evidence_summary=[
                    (
                        f"span duration p95 increased from {baseline_p95:.2f} to {incident_p95:.2f} "
                        f"baseline_points={baseline_points} incident_points={incident_points}"
                    )
                ],
                source_refs=[f"trace_service:{canonical}:duration_p95"],
            )
        )
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.canonical_target))


def _edge_frame(frame: pd.DataFrame, columns: TraceColumns) -> pd.DataFrame:
    if columns.caller is not None and columns.callee is not None:
        return pd.DataFrame(
            {
                "timestamp": frame[columns.timestamp],
                "caller": frame[columns.caller],
                "callee": frame[columns.callee],
                "duration": frame[columns.duration],
            }
        )
    required = [columns.trace_id, columns.span_id, columns.parent_span_id]
    if any(column is None for column in required):
        return pd.DataFrame(columns=["timestamp", "caller", "callee", "duration"])

    work = frame[
        [
            columns.timestamp,
            columns.service,
            columns.duration,
            columns.trace_id,
            columns.span_id,
            columns.parent_span_id,
        ]
    ].copy()
    work.columns = [
        "timestamp",
        "service",
        "duration",
        "trace_id",
        "span_id",
        "parent_span_id",
    ]
    parents = work[["trace_id", "span_id", "service"]].rename(
        columns={"span_id": "parent_span_id", "service": "caller"}
    )
    joined = work.merge(parents, on=["trace_id", "parent_span_id"], how="left")
    joined = joined[joined["caller"].notna()]
    return pd.DataFrame(
        {
            "timestamp": joined["timestamp"],
            "caller": joined["caller"],
            "callee": joined["service"],
            "duration": joined["duration"],
        }
    )


def _rank_shadow_edges(
    edges: list[ShadowEdgeCandidate],
) -> list[ShadowEdgeCandidate]:
    ordered = sorted(edges, key=lambda edge: (-edge.source_score, edge.caller, edge.callee))
    return [
        edge.model_copy(update={"rank": rank})
        for rank, edge in enumerate(ordered, start=1)
    ]


def _build_edge_candidates(
    case_id: str,
    frame: pd.DataFrame,
    columns: TraceColumns,
    windows: IncidentWindows,
) -> tuple[list[SourceCandidate], list[ShadowEdgeCandidate]]:
    edges = _edge_frame(frame, columns)
    if edges.empty:
        return [], []

    baseline_mask, incident_mask = _window_masks(edges, "timestamp", windows)
    projected: list[SourceCandidate] = []
    shadow_edges: list[ShadowEdgeCandidate] = []
    for (raw_caller, raw_callee), group in edges.groupby(
        ["caller", "callee"], dropna=True
    ):
        caller = canonicalize_component(str(raw_caller))
        callee = canonicalize_component(str(raw_callee))
        if not caller or not callee or caller == callee:
            continue
        baseline = group.loc[
            baseline_mask.reindex(group.index, fill_value=False), "duration"
        ]
        incident = group.loc[
            incident_mask.reindex(group.index, fill_value=False), "duration"
        ]
        score, baseline_p95, incident_p95, baseline_points, incident_points = (
            _score_duration_shift(baseline, incident)
        )
        if score <= 0.0:
            continue
        edge_key = f"edge:{caller}->{callee}"
        summary = (
            f"edge latency p95 increased from {baseline_p95:.2f} to {incident_p95:.2f} "
            f"baseline_points={baseline_points} incident_points={incident_points}"
        )
        shadow_edges.append(
            ShadowEdgeCandidate(
                case_id=case_id,
                edge_key=edge_key,
                caller=caller,
                callee=callee,
                raw_caller=str(raw_caller),
                raw_callee=str(raw_callee),
                source_score=score,
                evidence_summary=[summary],
                source_refs=[f"trace_edge:{caller}->{callee}:duration_p95"],
                rank=1,
            )
        )
        for service, raw_service in ((callee, raw_callee), (caller, raw_caller)):
            projected.append(
                SourceCandidate(
                    case_id=case_id,
                    candidate_key=f"service:{service}",
                    target_type="service",
                    canonical_target=service,
                    raw_target=str(raw_service),
                    source="trace",
                    source_bucket="trace_edge_projected_service",
                    score=round(score * 0.75, 6),
                    evidence_summary=[
                        f"projected from {edge_key}: {summary}"
                    ],
                    source_refs=[
                        f"trace_edge:{caller}->{callee}:projected_service:{service}"
                    ],
                    projection_from_edge=edge_key,
                )
            )
    projected = sorted(
        projected, key=lambda candidate: (-candidate.score, candidate.canonical_target)
    )
    return projected, _rank_shadow_edges(shadow_edges)


def build_trace_candidates_from_dataframe(
    case_id: str,
    traces: pd.DataFrame,
    windows: IncidentWindows,
) -> TraceBuildResult:
    columns = _detect_columns(traces)
    if columns is None:
        return TraceBuildResult(
            case_id=case_id,
            source_candidates=[],
            shadow_edges=[],
            warnings=["missing required trace columns: timestamp, service, duration"],
        )

    frame = traces.copy()
    frame[columns.timestamp] = pd.to_numeric(frame[columns.timestamp], errors="coerce")
    frame[columns.duration] = pd.to_numeric(frame[columns.duration], errors="coerce")
    frame = frame.dropna(subset=[columns.timestamp, columns.duration, columns.service])

    service_candidates = _build_service_candidates(case_id, frame, columns, windows)
    projected_candidates, shadow_edges = _build_edge_candidates(
        case_id, frame, columns, windows
    )
    return TraceBuildResult(
        case_id=case_id,
        source_candidates=service_candidates + projected_candidates,
        shadow_edges=shadow_edges,
        warnings=[],
    )


def build_trace_candidates_for_case(
    case_id: str,
    traces_path: Path,
    windows: IncidentWindows,
) -> TraceBuildResult:
    traces = pd.read_csv(traces_path)
    return build_trace_candidates_from_dataframe(case_id, traces, windows)
