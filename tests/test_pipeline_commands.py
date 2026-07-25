"""Tests for the `pipeline` command group (compose, list, run).

Tests the Pipeline module class construction and the 3 CLI subcommands
without requiring a real LM (the LM is auto-configured by setup_dspy()
to a default mock and cached inference suppresses actual API calls).
"""

from __future__ import annotations

from click.testing import CliRunner

from dspytools.commands.pipeline import pipeline_cmd
from dspytools.core.registry import save_run_index

# ═══════════════════════════════════════════════════════════════════════════
# 1. Pipeline class construction
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineConstruction:
    """The inline Pipeline class creates the right number of stages."""

    def _make_pipeline(self, module_names: list[str]):
        from dspytools.core._dspy import dspy

        class Pipeline(dspy.Module):
            def __init__(self):
                super().__init__()
                self.stages = []
                for _mod_name in module_names:
                    self.stages.append(dspy.Predict("input -> output"))

            def forward(self, input: str) -> dspy.Prediction:
                x = input
                for stage in self.stages:
                    result = stage(input=x)
                    x = getattr(result, "output", str(result))
                return dspy.Prediction(output=x)

        return Pipeline()

    def test_two_stages(self):
        """2 modules → 2 stages."""
        p = self._make_pipeline(["mod_a", "mod_b"])
        assert len(p.stages) == 2

    def test_three_stages(self):
        """3 modules → 3 stages."""
        p = self._make_pipeline(["a", "b", "c"])
        assert len(p.stages) == 3

    def test_one_stage(self):
        """1 module (below CLI minimum) still constructs fine."""
        p = self._make_pipeline(["single"])
        assert len(p.stages) == 1

    def test_five_stages(self):
        """5 modules → 5 stages."""
        p = self._make_pipeline(["m1", "m2", "m3", "m4", "m5"])
        assert len(p.stages) == 5

    def test_stages_are_distinct_objects(self):
        """Each stage is a separate Predict instance."""
        p = self._make_pipeline(["a", "b"])
        assert p.stages[0] is not p.stages[1]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pipeline class forward chaining
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineForward:
    """Pipeline.forward() chains outputs properly."""

    def test_forward_returns_prediction(self):
        """forward() returns a dspy.Prediction with 'output' field."""
        from dspytools.core._dspy import dspy

        # DSPy is already configured by the test suite (no-op on re-entry).
        from dspytools.core.setup import setup_dspy

        setup_dspy()

        class Pipeline(dspy.Module):
            def __init__(self):
                super().__init__()
                self.stages = [dspy.Predict("input -> output")]

            def forward(self, input: str) -> dspy.Prediction:
                result = self.stages[0](input=input)
                return dspy.Prediction(output=getattr(result, "output", str(result)))

        p = Pipeline()
        result = p(input="test query")
        assert hasattr(result, "output")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLI compose — valid usage
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineComposeCLI:
    """dspytools pipeline compose <modules...>"""

    def setup_method(self):
        save_run_index([])
        self.runner = CliRunner()

    def teardown_method(self):
        save_run_index([])

    def test_compose_two_modules(self):
        """2 modules creates a pipeline successfully."""
        result = self.runner.invoke(
            pipeline_cmd, ["compose", "mod1", "mod2", "-n", "test-pipe"]
        )
        assert result.exit_code == 0
        assert "Pipeline" in result.output
        assert "test-pipe" in result.output
        assert "2 stages" in result.output

    def test_compose_three_modules(self):
        """3 modules creates a pipeline."""
        result = self.runner.invoke(
            pipeline_cmd, ["compose", "a", "b", "c", "-n", "three-stage"]
        )
        assert result.exit_code == 0
        assert "3 stages" in result.output

    def test_compose_registers_run(self):
        """Compose registers pipeline in the compiled run registry."""
        from dspytools.core.registry import list_compiled_runs

        self.runner.invoke(pipeline_cmd, ["compose", "m1", "m2", "-n", "reg-test"])
        runs = list_compiled_runs()
        pipe_runs = [r for r in runs if r.get("type") == "pipeline"]
        assert len(pipe_runs) >= 1
        assert pipe_runs[0]["name"] == "reg-test"
        assert pipe_runs[0]["modules"] == ["m1", "m2"]

    def test_compose_with_optional_label(self):
        """--label flag is accepted."""
        result = self.runner.invoke(
            pipeline_cmd,
            ["compose", "mod1", "mod2", "-n", "labeled", "--label", "v1"],
        )
        assert result.exit_code == 0

    def test_compose_default_name(self):
        """No --name flag defaults to 'pipeline'."""
        result = self.runner.invoke(pipeline_cmd, ["compose", "mod1", "mod2"])
        assert result.exit_code == 0
        assert "pipeline" in result.output

    def test_compose_with_many_modules(self):
        """10 modules is accepted."""
        modules = [f"m{i}" for i in range(10)]
        result = self.runner.invoke(
            pipeline_cmd, ["compose"] + modules + ["-n", "ten-stage"]
        )
        assert result.exit_code == 0
        assert "10 stages" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# 4. CLI compose — error cases
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineComposeErrors:
    """dspytools pipeline compose — validation."""

    def setup_method(self):
        save_run_index([])
        self.runner = CliRunner()

    def teardown_method(self):
        save_run_index([])

    def test_compose_one_module_fails(self):
        """1 module → error."""
        result = self.runner.invoke(pipeline_cmd, ["compose", "single"])
        assert result.exit_code != 0
        assert "at least 2 modules" in result.output.lower()

    def test_compose_zero_modules_fails(self):
        """0 modules → error."""
        result = self.runner.invoke(pipeline_cmd, ["compose"])
        assert result.exit_code != 0
        assert "at least 2 modules" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI list
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineListCLI:
    """dspytools pipeline list"""

    def setup_method(self):
        save_run_index([])
        self.runner = CliRunner()

    def teardown_method(self):
        save_run_index([])

    def test_list_empty(self):
        """No pipelines → message."""
        result = self.runner.invoke(pipeline_cmd, ["list"])
        assert result.exit_code == 0
        assert "No pipelines found" in result.output

    def test_list_one_pipeline(self):
        """One pipeline appears in the table."""
        self.runner.invoke(pipeline_cmd, ["compose", "m1", "m2", "-n", "pipe-one"])
        result = self.runner.invoke(pipeline_cmd, ["list"])
        assert result.exit_code == 0
        assert "pipe-one" in result.output
        assert "ID" in result.output
        assert "Modules" in result.output

    def test_list_multiple_pipelines(self):
        """Multiple pipelines all appear."""
        self.runner.invoke(pipeline_cmd, ["compose", "a", "b", "-n", "pipe-a"])
        self.runner.invoke(pipeline_cmd, ["compose", "x", "y", "z", "-n", "pipe-b"])
        result = self.runner.invoke(pipeline_cmd, ["list"])
        assert result.exit_code == 0
        assert "pipe-a" in result.output
        assert "pipe-b" in result.output

    def test_list_does_not_show_non_pipeline_runs(self):
        """Only type=pipeline runs are listed."""
        from dspytools.core.registry import register_run

        register_run("compile_run_1", {"optimizer": "mipro", "type": "compile"})
        register_run(
            "pipeline_run_1",
            {"name": "real-pipe", "modules": ["x", "y"], "type": "pipeline"},
        )
        result = self.runner.invoke(pipeline_cmd, ["list"])
        assert "real-pipe" in result.output
        # Non-pipeline runs should not appear
        assert "compile_run_1" not in result.output


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI run
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineRunCLI:
    """dspytools pipeline run <id> --input <text>"""

    def setup_method(self):
        save_run_index([])
        self.runner = CliRunner()

    def teardown_method(self):
        save_run_index([])

    def _compose_and_get_id(self, name: str = "pipe-run") -> str:
        """Helper: compose a pipeline and return its run_id."""
        self.runner.invoke(pipeline_cmd, ["compose", "m1", "m2", "-n", name])
        from dspytools.core.registry import list_compiled_runs

        runs = [r for r in list_compiled_runs() if r.get("type") == "pipeline"]
        assert len(runs) >= 1
        return runs[0]["id"]

    def test_run_with_composed_pipeline(self):
        """Running a composed pipeline returns output."""
        run_id = self._compose_and_get_id("test-run")
        result = self.runner.invoke(
            pipeline_cmd, ["run", run_id, "--input", "hello world"]
        )
        assert result.exit_code == 0
        assert "Input:" in result.output
        assert "Output:" in result.output

    def test_run_nonexistent_pipeline(self):
        """Non-existent pipeline ID shows error."""
        result = self.runner.invoke(
            pipeline_cmd, ["run", "nonexistent", "--input", "test"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_run_without_input(self):
        """Missing --input flag shows usage error."""
        result = self.runner.invoke(pipeline_cmd, ["run", "some-id"])
        assert result.exit_code != 0
        # Click reports missing option
        assert "--input" in result.output or "option" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Help output
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineHelp:
    """dspytools pipeline --help"""

    def test_pipeline_help(self):
        """--help shows available subcommands."""
        runner = CliRunner()
        result = runner.invoke(pipeline_cmd, ["--help"])
        assert result.exit_code == 0
        assert "compose" in result.output
        assert "list" in result.output
        assert "run" in result.output

    def test_compose_help(self):
        """compose --help shows usage."""
        runner = CliRunner()
        result = runner.invoke(pipeline_cmd, ["compose", "--help"])
        assert result.exit_code == 0
        assert "modules" in result.output.lower()

    def test_run_help(self):
        """run --help shows --input option."""
        runner = CliRunner()
        result = runner.invoke(pipeline_cmd, ["run", "--help"])
        assert result.exit_code == 0
        assert "input" in result.output.lower()
