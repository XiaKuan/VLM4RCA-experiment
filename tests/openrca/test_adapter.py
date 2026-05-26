import json
from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter


def _write_case(root: Path, case_id: str, *, include_logs: bool = True) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "matched_faults": [
                    {"component": "Tomcat01", "reason": "network latency", "timestamp": 1000}
                ],
                "system_components": ["Tomcat01"],
                "evidence_components": ["Tomcat01"],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.csv").write_text("timestamp,Tomcat01__cpu\n940,1\n", encoding="utf-8")
    (case_dir / "traces.csv").write_text(
        "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n940,940,Tomcat01,t,s,p,1\n",
        encoding="utf-8",
    )
    if include_logs:
        (case_dir / "logs.csv").write_text(
            "time,timestamp,service,log_name,message\n940,940,Tomcat01,app,ok\n", encoding="utf-8"
        )


def test_bank_adapter_lists_cases_and_loads_metadata(tmp_path: Path) -> None:
    _write_case(tmp_path, "task3_002")
    _write_case(tmp_path, "task3_001")

    adapter = OpenRCABankAdapter(tmp_path)

    assert adapter.list_case_ids() == ["task3_001", "task3_002"]
    assert adapter.case_path("task3_001") == tmp_path / "cases" / "task3_001"
    assert adapter.load_case_meta("task3_001")["matched_faults"][0]["component"] == "Tomcat01"


def test_bank_adapter_reports_existing_non_empty_telemetry(tmp_path: Path) -> None:
    _write_case(tmp_path, "task3_001", include_logs=False)
    (tmp_path / "cases" / "task3_001" / "logs.csv").write_text("", encoding="utf-8")

    adapter = OpenRCABankAdapter(tmp_path)

    assert adapter.telemetry_exists("task3_001", "metrics") is True
    assert adapter.telemetry_exists("task3_001", "traces") is True
    assert adapter.telemetry_exists("task3_001", "logs") is False
    assert adapter.telemetry_exists("task3_001", "topology") is False
