"""Test doctor diagnostics (Feature: dspytools doctor)."""

from click.testing import CliRunner

from dspytools.commands.doctor import doctor_cmd


def test_doctor_runs():
    """Doctor should run without crashing (check-llm and check-gpu disabled)."""
    runner = CliRunner()
    result = runner.invoke(doctor_cmd, ["--no-llm", "--no-gpu", "--no-config"])
    assert result.exit_code == 0
    assert "Python" in result.output or "python" in result.output.lower()


def test_doctor_exit_code_ok():
    """Doctor should exit 0 on healthy system (with all external checks disabled)."""
    runner = CliRunner()
    result = runner.invoke(doctor_cmd, ["--no-llm", "--no-gpu", "--no-config"])
    assert result.exit_code == 0
