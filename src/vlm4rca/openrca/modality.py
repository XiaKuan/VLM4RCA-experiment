from __future__ import annotations

from vlm4rca.openrca.adapter import OpenRCAAdapter
from vlm4rca.openrca.models import ModalityAvailability, TopologySource


def detect_modality_availability(adapter: OpenRCAAdapter, case_id: str) -> ModalityAvailability:
    metrics_available = adapter.telemetry_exists(case_id, "metrics")
    logs_available = adapter.telemetry_exists(case_id, "logs")
    traces_available = adapter.telemetry_exists(case_id, "traces")
    static_topology_available = adapter.telemetry_exists(case_id, "topology")

    if static_topology_available:
        topology_available = True
        topology_source: TopologySource = "static_topology"
    elif traces_available:
        topology_available = True
        topology_source = "trace_derived"
    else:
        topology_available = False
        topology_source = "unavailable"

    warnings: list[str] = []
    if not metrics_available:
        warnings.append("metrics.csv is missing or empty")
    if not logs_available:
        warnings.append("logs.csv is missing or empty")
    if not traces_available:
        warnings.append("traces.csv is missing or empty")

    return ModalityAvailability(
        metrics_available=metrics_available,
        traces_available=traces_available,
        logs_available=logs_available,
        topology_available=topology_available,
        topology_source=topology_source,
        warnings=warnings,
    )
