from vlm4rca.openrca.windows import WindowConfig, extract_incident_windows


def test_extracts_default_phase1_windows_from_inject_time() -> None:
    windows = extract_incident_windows(
        {
            "inject_time": 1_614_841_020,
            "context_start": 1_614_835_800,
            "context_end": 1_614_843_000,
        }
    )

    assert windows.baseline_start == 1_614_838_620
    assert windows.baseline_end == 1_614_840_420
    assert windows.incident_start == 1_614_840_420
    assert windows.incident_end == 1_614_842_220
    assert windows.boundary_rule == "[start, end)"
    assert windows.timezone == "UTC"
    assert windows.warnings == []


def test_records_context_coverage_warnings() -> None:
    windows = extract_incident_windows(
        {
            "inject_time": 1_000,
            "context_start": 100,
            "context_end": 1_100,
        },
        WindowConfig(pre_window_seconds=100, post_window_seconds=200, baseline_window_seconds=300),
    )

    assert windows.baseline_start == 600
    assert windows.baseline_end == 900
    assert windows.incident_start == 900
    assert windows.incident_end == 1_200
    assert windows.warnings == ["context_end 1100 is before incident_end 1200"]
