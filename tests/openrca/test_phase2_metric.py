import json
from pathlib import Path

import pandas as pd

from vlm4rca.candidates.models import VariantRecallSummary
from vlm4rca.openrca.phase2_metric import (
    build_metric_baseline_report_markdown,
    main,
    run_metric_baseline,
)


def _write_case(root: Path, case_id: str, component: str, hit_component: str) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1500,
                "context_start": 0,
                "context_end": 3000,
                "matched_faults": [
                    {"component": component, "reason": "high CPU usage", "timestamp": 1000}
                ],
                "system_components": [component, hit_component, "Redis01"],
                "evidence_components": [component, hit_component, "Redis01"],
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1000, 1100],
            f"{hit_component}__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 40, 41, 42],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [20, 20, 20, 20, 20, 20],
        }
    )
    frame.to_csv(case_dir / "metrics.csv", index=False)
    (case_dir / "logs.csv").write_text(
        "time,timestamp,service,log_name,message\n1,1,a,app,ok\n", encoding="utf-8"
    )
    (case_dir / "traces.csv").write_text(
        "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n1,1,a,t,s,p,2\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, data_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                f"data_root: {data_root.as_posix()}",
                "cases:",
                "  - case_id: case_1",
                "    case_group: metric_obvious",
                "    selection_reason: fixture one",
                "  - case_id: case_2",
                "    case_group: soft_latency",
                "    selection_reason: fixture two",
            ]
        ),
        encoding="utf-8",
    )


def test_metric_baseline_report_markdown_contains_first_m_row() -> None:
    markdown = build_metric_baseline_report_markdown(
        VariantRecallSummary(
            variant="M",
            recall_at_3=0.5,
            recall_at_5=0.5,
            recall_at_8=1.0,
            soft_recall_at_8=0.0,
            avg_candidates=4.25,
        )
    )

    assert (
        "| Variant | Recall@3 | Recall@5 | Recall@8 | Soft Recall@8 | Avg Candidates |" in markdown
    )
    assert "| M | 0.500 | 0.500 | 1.000 | 0.000 | 4.25 |" in markdown


def test_run_metric_baseline_writes_candidates_recall_and_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", hit_component="Tomcat01")
    _write_case(data_root, "case_2", component="Mysql02", hit_component="Redis01")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "outputs"

    result = run_metric_baseline(manifest_path, data_root, output_dir)

    assert result["report_row"]["variant"] == "M"
    assert result["recall_summary"]["component_recall_at_k"]["3"] == 0.5
    assert (output_dir / "candidates" / "M" / "case_1.json").exists()
    assert (output_dir / "candidates" / "M" / "case_2.json").exists()
    assert (output_dir / "recall_M.json").exists()
    assert (output_dir / "baseline_report.md").exists()

    candidate_payload = json.loads(
        (output_dir / "candidates" / "M" / "case_1.json").read_text(encoding="utf-8")
    )
    assert len(candidate_payload["candidates"]) <= 8
    assert candidate_payload["candidates"][0]["candidate_key"] == "service:tomcat01"
    assert (
        candidate_payload["candidates"][0]["variant_candidate_id"]
        == "cand:case_1:M:1:service:tomcat01"
    )


def test_phase2_metric_cli_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", hit_component="Tomcat01")
    _write_case(data_root, "case_2", component="Mysql02", hit_component="Redis01")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase2"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "recall_M.json").read_text(encoding="utf-8"))
    assert summary["component_recall_at_k"]["8"] == 0.5
    assert "Metric-only Baseline Report" in (output_dir / "baseline_report.md").read_text(
        encoding="utf-8"
    )


def test_real_pilot15_metric_baseline_acceptance(tmp_path: Path) -> None:
    manifest_path = Path("configs/openrca_pilot15.yaml")
    data_root = Path("data/OpenRCA/Bank")
    if not manifest_path.exists() or not data_root.exists():
        return

    result = run_metric_baseline(
        manifest_path=manifest_path,
        data_root=data_root,
        output_dir=tmp_path / "phase2_metric",
    )

    metric_results = result["metric_results"]
    assert len(metric_results) == 15
    for case_id, payload in metric_results.items():
        candidates = payload["candidates"]
        assert len(candidates) <= 8
        for candidate in candidates:
            assert (
                candidate["candidate_key"]
                == f"{candidate['target_type']}:{candidate['canonical_target']}"
            )
            assert candidate["variant_candidate_id"] == (
                f"cand:{case_id}:M:{candidate['rank']}:{candidate['target_type']}:{candidate['canonical_target']}"
            )
            assert candidate["present_in_variants"] == ["M"]
            assert candidate["sources"] == ["metric"]

    report_row = result["report_row"]
    assert report_row["variant"] == "M"
    assert 0.0 <= report_row["recall_at_3"] <= 1.0
    assert 0.0 <= report_row["recall_at_5"] <= 1.0
    assert 0.0 <= report_row["recall_at_8"] <= 1.0
    assert report_row["avg_candidates"] <= 8.0
