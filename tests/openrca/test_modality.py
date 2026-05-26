from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.modality import detect_modality_availability


def _write_case_files(case_dir: Path, *, traces: bool = True, topology: bool = False) -> None:
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        '{"case_id": "case_1", "inject_time": 1000}', encoding="utf-8"
    )
    (case_dir / "metrics.csv").write_text("timestamp,cpu\n1,2\n", encoding="utf-8")
    (case_dir / "logs.csv").write_text(
        "time,timestamp,service,log_name,message\n1,1,a,app,ok\n", encoding="utf-8"
    )
    if traces:
        (case_dir / "traces.csv").write_text(
            "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n1,1,a,t,s,p,2\n",
            encoding="utf-8",
        )
    if topology:
        (case_dir / "topology.csv").write_text("source,target\na,b\n", encoding="utf-8")


def test_detects_trace_derived_topology_when_traces_exist(tmp_path: Path) -> None:
    _write_case_files(tmp_path / "cases" / "case_1", traces=True)
    adapter = OpenRCABankAdapter(tmp_path)

    availability = detect_modality_availability(adapter, "case_1")

    assert availability.metrics_available is True
    assert availability.logs_available is True
    assert availability.traces_available is True
    assert availability.topology_available is True
    assert availability.topology_source == "trace_derived"
    assert availability.warnings == []


def test_static_topology_is_available_without_traces(tmp_path: Path) -> None:
    _write_case_files(tmp_path / "cases" / "case_1", traces=False, topology=True)
    adapter = OpenRCABankAdapter(tmp_path)

    availability = detect_modality_availability(adapter, "case_1")

    assert availability.traces_available is False
    assert availability.topology_available is True
    assert availability.topology_source == "static_topology"
