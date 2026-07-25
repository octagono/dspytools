"""Test drift detection (Feature #2)."""

import tempfile
from pathlib import Path

from dspytools.core.drift_monitor import DriftMonitor


def _isolated_monitor() -> DriftMonitor:
    """Create a DriftMonitor with a fresh temp state file."""
    return DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "test_drift.json"))


def test_drift_warning():
    m = _isolated_monitor()
    m.update_baseline("test_prog", 0.85)
    alert = m.check("test_prog", 0.79)
    assert alert is not None
    assert alert.severity == "warning"
    assert alert.baseline_score == 0.85
    assert alert.current_score == 0.79


def test_drift_critical():
    m = _isolated_monitor()
    m.update_baseline("test_prog", 0.85)
    alert = m.check("test_prog", 0.70)
    assert alert is not None
    assert alert.severity == "critical"


def test_drift_no_alert():
    m = _isolated_monitor()
    m.update_baseline("test_prog", 0.85)
    alert = m.check("test_prog", 0.84)
    assert alert is None


def test_drift_history():
    m = _isolated_monitor()
    m.update_baseline("test_prog", 0.85)
    m.check("test_prog", 0.83)
    m.check("test_prog", 0.81)

    history = m.get_history("test_prog")
    assert len(history) == 2
    assert "score" in history[0]
    assert "delta" in history[0]


def test_drift_status():
    m = _isolated_monitor()
    m.update_baseline("prog_a", 0.85)
    m.update_baseline("prog_b", 0.90)

    status = m.status
    assert status["programs_tracked"] == 2
    assert "prog_a" in status["programs"]
    assert status["thresholds"]["warning"] == 0.05


def test_drift_unknown_program():
    m = _isolated_monitor()
    alert = m.check("nonexistent", 0.50)
    assert alert is None


# ── Edge case tests ──────────────────────────────────────────────────────


def test_drift_first_run():
    """First run with no baseline should return None (no alert)."""
    m = DriftMonitor()
    alert = m.check("new_prog", 0.50)
    assert alert is None


def test_drift_repeated_degradation():
    """Multiple checks with degradation accumulate consecutive drops."""
    import tempfile

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "test_drift2.json"))
    m.update_baseline("prog", 0.90)
    for i in range(5):
        score = 0.90 - (i + 1) * 0.02
        m.check("prog", score)
    history = m.get_history("prog")
    assert len(history) == 5
    assert history[-1]["score"] < history[0]["score"]


def test_drift_recovery():
    """Score recovering above baseline should not trigger alert."""
    m = DriftMonitor()
    m.update_baseline("prog", 0.80)
    # Degrade
    m.check("prog", 0.72)
    # Recover
    alert = m.check("prog", 0.85)
    assert alert is None


def test_drift_baseline_update():
    """Updating baseline resets the reference point."""
    m = DriftMonitor()
    m.update_baseline("prog", 0.80)
    m.check("prog", 0.75)
    m.update_baseline("prog", 0.90)
    alert = m.check("prog", 0.85)
    # 0.85 vs 0.90 = ~5.5% degradation → should be a warning
    assert alert is not None


def test_drift_history_max_size():
    """History is capped at 50 entries."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "test_drift3.json"))
    m.update_baseline("prog", 0.90)
    for i in range(60):
        score = 0.90 - (i % 3) * 0.05
        m.check("prog", score)
    history = m.get_history("prog")
    assert len(history) <= 50


def test_drift_status_empty():
    """Status with no baselines shows 0 tracked."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "fresh_drift.json"))
    status = m.status
    assert status["programs_tracked"] == 0
    assert status["programs"] == {}


def test_drift_exact_threshold():
    """Degradation exactly at threshold (5%) should still trigger warning."""
    m = DriftMonitor()
    m.update_baseline("prog", 1.0)
    # 0.95 vs 1.0 = exactly 5% degradation = threshold
    # degradation_pct = (0.05/1.0) * 100 = 5.0
    alert = m.check("prog", 0.95)
    assert alert is not None
    assert alert.severity == "warning"


def test_drift_just_below_threshold():
    """Degradation just below 5% threshold should not trigger."""
    m = DriftMonitor()
    m.update_baseline("prog", 1.0)
    alert = m.check("prog", 0.951)
    delta = 1.0 - 0.951
    degradation_pct = (delta / 1.0) * 100
    if degradation_pct >= 5.0:
        assert alert is not None
    else:
        assert alert is None


# ── Recompile request tests ──────────────────────────────────────────────


def test_recompile_request():
    """request_recompile queues a program for recompilation."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "drift_recompile.json"))
    assert m.pending_recompiles() == []
    m.request_recompile("prog_a")
    assert "prog_a" in m.pending_recompiles()


def test_recompile_clear():
    """clear_recompile_request removes a queued program."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "drift_recompile2.json"))
    m.request_recompile("prog_a")
    m.request_recompile("prog_b")
    assert len(m.pending_recompiles()) == 2
    m.clear_recompile_request("prog_a")
    assert "prog_a" not in m.pending_recompiles()
    assert "prog_b" in m.pending_recompiles()


def test_recompile_process_dry_run():
    """process_recompile_requests with auto_fix=False reports pending."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "drift_recompile3.json"))
    m.request_recompile("prog_a")
    results = m.process_recompile_requests(auto_fix=False)
    assert len(results) == 1
    assert results[0]["status"] == "pending"
    assert "prog_a" in m.pending_recompiles()


def test_recompile_no_pending():
    """process_recompile_requests with no pending returns empty."""
    import tempfile
    from pathlib import Path

    m = DriftMonitor(state_file=str(Path(tempfile.mkdtemp()) / "drift_recompile4.json"))
    assert m.process_recompile_requests() == []
