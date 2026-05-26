from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlm4rca.openrca.phase3_multisource import (
    build_candidate_recall_markdown,
    main,
    run_multisource_candidate_recall,
)


def _write_case(root: Path, case_id: str, component: str, metric_hit: str, trace_hit: str) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "context_start": 0,
                "context_end": 2000,
                "matched_faults": [{"component": component, "reason": "latency", "timestamp": 1000}],
                "system_components": [component, metric_hit, trace_hit, "Gateway", "Payment"],
                "evidence_components": [component, metric_hit, trace_hit, "Gateway", "Payment"],
            }
        ),
        encoding="utf-8",
    )
    # inject_time=1000 → baseline=[-1400,400), incident=[400,2200)
    # timestamps: 100,200,300 in baseline; 600,700,800 in incident
    pd.DataFrame(
        {
            "timestamp": [100, 200, 300, 600, 700, 800],
            f"{metric_hit}__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 40, 41, 42],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [20, 20, 20, 20, 20, 20],
        }
    ).to_csv(case_dir / "metrics.csv", index=False)
    # traces: baseline durations low, incident durations high for trace_hit service
    pd.DataFrame(
        {
            "timestamp": [100, 200, 300, 600, 700, 800],
            "trace_id": ["a", "b", "c", "d", "e", "f"],
            "span_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "parent_span_id": ["", "", "", "", "", ""],
            "service": [trace_hit, trace_hit, trace_hit, trace_hit, trace_hit, trace_hit],
            "caller": ["Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Gateway"],
            "callee": [trace_hit, trace_hit, trace_hit, trace_hit, trace_hit, trace_hit],
            "duration": [5, 5, 5, 30, 31, 32],
        }
    ).to_csv(case_dir / "traces.csv", index=False)
    # logs: mix of baseline ok and incident timeout messages
    pd.DataFrame(
        {
            "timestamp": [110, 220, 610, 730, 750],
            "service": [trace_hit, trace_hit, trace_hit, trace_hit, trace_hit],
            "message": ["ok", "ok", "timeout calling downstream", "retry backoff", "error connection"],
        }
    ).to_csv(case_dir / "logs.csv", index=False)


def _write_manifest(path: Path, data_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                f"data_root: {data_root.as_posix()}",
                "cases:",
                "  - case_id: case_1",
                "    case_group: metric_obvious",
                "    selection_reason: metric hit fixture",
                "  - case_id: case_2",
                "    case_group: soft_latency",
                "    selection_reason: trace recovery fixture",
            ]
        ),
        encoding="utf-8",
    )


def test_candidate_recall_markdown_contains_primary_and_secondary_tables() -> None:
    markdown = build_candidate_recall_markdown(
        {
            "primary": {
                "M": {"recall_at_3": 0.5, "recall_at_5": 0.5, "recall_at_8": 0.5, "soft_recall_at_8": 0.0, "avg_candidates": 1.0},
                "M+T": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 2.0},
                "M+T+L": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 2.0},
                "M+T+L+Topo": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 3.0},
            },
            "secondary": {
                "M": {"recall_at_3": 0.5, "recall_at_5": 0.5, "recall_at_8": 0.5, "soft_recall_at_8": 0.0, "avg_candidates": 1.0},
                "M+T": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 2.0},
                "M+T+L": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 2.0},
                "M+T+L+Topo": {"recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_8": 1.0, "soft_recall_at_8": 1.0, "avg_candidates": 3.0},
            },
        }
    )

    assert "Primary Analysis" in markdown
    assert "Secondary Analysis" in markdown
    assert "| Full | 1.000 | 1.000 | 1.000 | 1.000 | 3.00 |" in markdown


def test_run_multisource_candidate_recall_writes_all_variants_and_checkpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", metric_hit="Tomcat01", trace_hit="Tomcat01")
    _write_case(data_root, "case_2", component="Payment", metric_hit="Redis01", trace_hit="Payment")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase3"

    result = run_multisource_candidate_recall(manifest_path, data_root, output_dir)

    assert result["checkpoint_decision"]["metric_recall_at_8"] == 0.5
    assert result["checkpoint_decision"]["full_recall_at_8"] == 1.0
    assert result["checkpoint_decision"]["proceed_to_vlm_and_rendering"] is True
    for variant in ["M", "M+T", "M+T+L", "M+T+L+Topo"]:
        assert (output_dir / "candidates" / variant / "case_1.json").exists()
        assert (output_dir / "candidates" / variant / "case_2.json").exists()
        payload = json.loads((output_dir / "candidates" / variant / "case_2.json").read_text(encoding="utf-8"))
        assert len(payload["candidates"]) <= 8
    first_hit = json.loads((output_dir / "first_hit_sources.json").read_text(encoding="utf-8"))
    assert first_hit[1]["first_hit_source"] == "Trace"
    assert first_hit[1]["new_hit_source"] == "Trace"
    assert (output_dir / "candidate_recall_ablation.md").exists()
    assert (output_dir / "shadow_edge_metrics.json").exists()


def test_phase3_cli_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", metric_hit="Tomcat01", trace_hit="Tomcat01")
    _write_case(data_root, "case_2", component="Payment", metric_hit="Redis01", trace_hit="Payment")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase3_cli"

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
    report = json.loads((output_dir / "candidate_recall_ablation.json").read_text(encoding="utf-8"))
    assert report["secondary"]["M"]["recall_at_8"] == 0.5
    assert report["secondary"]["M+T"]["recall_at_8"] == 1.0


def test_real_pilot15_multisource_candidate_recall_acceptance(tmp_path: Path) -> None:
    manifest_path = Path("configs/openrca_pilot15.yaml")
    data_root = Path("data/OpenRCA/Bank")
    if not manifest_path.exists() or not data_root.exists():
        return

    result = run_multisource_candidate_recall(
        manifest_path=manifest_path,
        data_root=data_root,
        output_dir=tmp_path / "phase3_multisource",
    )

    report = result["ablation"]
    assert set(report.keys()) == {"primary", "secondary"}
    for section in ("primary", "secondary"):
        assert list(report[section].keys()) == ["M", "M+T", "M+T+L", "M+T+L+Topo"]
        for variant in ("M", "M+T", "M+T+L", "M+T+L+Topo"):
            row = report[section][variant]
            assert 0.0 <= row["recall_at_3"] <= 1.0
            assert 0.0 <= row["recall_at_5"] <= 1.0
            assert 0.0 <= row["recall_at_8"] <= 1.0
            assert row["avg_candidates"] <= 8.0
    assert len(result["first_hit_rows"]) == 15
    assert result["checkpoint_decision"]["eligible_cases"] >= 0
    assert result["shadow_edge_metrics"]["diagnostic_only"] is True
