"""dspytools generate — llms.txt generation and repository analysis.

Commands:
    llms-txt     Generate llms.txt for a repository
    batch        Run batch evaluation on devset
    explore      Deep repo analysis via MCP git tools
"""

from __future__ import annotations

from pathlib import Path

from dspytools.cli.output import console, info, label_option, ok, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core._dspy import dspy
from dspytools.core.logging_config import get_logger
from dspytools.core.output import create_run_dir, save_program
from dspytools.core.registry import register_run
from dspytools.core.setup import LMRegistry, setup_dspy

_log = get_logger(__name__)


@click.group(name="generate", cls=LLMGroup)
def generate_cmd():
    """Generate llms.txt documentation for repositories."""


# ── llms-txt ────────────────────────────────────────────────────────────


@generate_cmd.command(name="llms-txt", cls=LLMCommand)
@click.argument("target")
@click.option("--local/--remote", default=False, help="Target is a local directory")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--baml/--no-baml", default=False, help="Use BAML adapter")
@label_option
def gen_llms_txt(
    target: str,
    local: bool,
    output: str | None,
    baml: bool,
    label: str | None,
):
    """Generate llms.txt for a repository.

    TARGET can be a GitHub URL or local directory path.

    \b
    Examples:
        dspytools generate llms-txt https://github.com/numpy/numpy
        dspytools generate llms-txt . --local
        dspytools generate llms-txt /path/to/repo --local --output repo.txt
    """
    from dspytools.generate import RepositoryAnalyzer, gather_repository_info

    setup_dspy()

    console.print(f"\n[bold]Generating llms.txt for: {target}[/]")

    # Gather repo info
    if local:
        repo_path = Path(target).resolve()
        if not repo_path.exists():
            raise click.ClickException(f"Path not found: {target}")
        file_tree, readme_content, package_files, history = gather_repository_info(
            str(repo_path)
        )
        info(f"Local repo: {repo_path.name} ({len(history.messages)} steps)")
    else:
        repo_url = target
        file_tree, readme_content, package_files, history = gather_repository_info()
        info(f"Simulated analysis: {len(history.messages)} steps")

    repo_url = target  # use target as repo_url
    analyzer = RepositoryAnalyzer()

    # Choose adapter
    adapter = None
    if baml:
        from dspy.adapters.baml_adapter import BAMLAdapter

        adapter = BAMLAdapter(use_native_function_calling=False)

    with dspy.context(adapter=adapter, temperature=0.3):
        result = analyzer(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content,
            package_files=package_files,
        )

    content = result.llms_txt_content
    _log.info("gen_llms_txt", target=target, chars=len(content))

    # Save output
    output_path = Path(output) if output else Path("generated_llms.txt")
    output_path.write_text(content)
    ok(f"Saved → {output_path} ({len(content)} chars)")

    # Register run
    run_id, run_path = create_run_dir("generate_llms_txt", label)
    save_program(
        run_path,
        analyzer,
        {
            "inputs": ["repo_url", "file_tree", "readme_content", "package_files"],
            "outputs": ["llms_txt_content"],
        },
        module_type="predict",
    )
    register_run(
        run_id,
        {"command": "generate_llms_txt", "target": target, "output": str(output_path)},
    )

    # Preview
    console.print(f"\n  Preview:\n{content[:500]}")


# ── batch ───────────────────────────────────────────────────────────────


@generate_cmd.command(name="batch", cls=LLMCommand)
@label_option
def gen_batch(label: str | None):
    """Run batch evaluation on devset.

    Evaluates the RepositoryAnalyzer module against the built-in devset
    using llms_txt_metric.
    """
    from dspytools.generate import (
        RepositoryAnalyzer,
        build_ground_truth_examples,
        llms_txt_metric,
    )

    setup_dspy()

    console.print("\n[bold]Batch Evaluation — llms.txt Quality[/]")

    _, devset = build_ground_truth_examples()
    analyzer = RepositoryAnalyzer()

    info(f"Devset: {len(devset)} examples")

    evaluator = dspy.Evaluate(
        devset=devset,
        metric=llms_txt_metric,
        num_threads=1,
        display_progress=True,
    )
    result = evaluator(analyzer)
    ok(f"Score: {result.score:.0%}")


# ── explore ─────────────────────────────────────────────────────────────


@generate_cmd.command(name="explore", cls=LLMCommand)
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, help="Output file path")
@label_option
def gen_explore(repo_path: str, output: str | None, label: str | None):
    """Deep repository analysis via MCP git tools + ReAct agent.

    Uses git-mcp tools to explore the repo, then generates llms.txt.
    Requires a working .mcp.json with git-mcp configured.

    Example:
        dspytools generate explore ./my-repo --output my-repo-llms.txt
    """
    from dspytools.generate import GitRepoExplorer, load_mcp_tools_sync

    setup_dspy()

    console.print(f"\n[bold]MCP Git Explorer: {repo_path}[/]")

    mcp_sessions, mcp_tools = load_mcp_tools_sync()

    if not mcp_tools:
        warn("No MCP tools loaded — check .mcp.json config")
        info("Expected: git-mcp via uvx in .mcp.json")
        raise click.Abort()

    info(f"Loaded {len(mcp_tools)} MCP tools")

    teacher = LMRegistry.get_teacher()
    explorer = GitRepoExplorer(mcp_tools=mcp_tools, teacher=teacher)

    info(f"Analyzing {Path(repo_path).name}...")
    result = explorer(repo_path=str(Path(repo_path).resolve()))

    _log.info("gen_explore", repo_path=repo_path, chars=len(result.llms_txt_content))
    ok(f"llms.txt: {len(result.llms_txt_content)} chars")
    info(f"Purpose: {result.purpose[:100]}...")

    output_path = Path(output) if output else Path("generated_llms_mcp.txt")
    output_path.write_text(result.llms_txt_content)
    ok(f"Saved → {output_path}")

    # Register
    run_id, run_path = create_run_dir("generate_explore", label)
    save_program(
        run_path,
        explorer,
        {"inputs": ["repo_path"], "outputs": ["llms_txt_content"]},
        module_type="agent",
    )
    register_run(
        run_id,
        {
            "command": "generate_explore",
            "repo_path": repo_path,
            "tools": len(mcp_tools),
        },
    )
    info(f"Run: {run_id}")


# ── warmup ─────────────────────────────────────────────────────────────


@generate_cmd.command(name="warmup", cls=LLMCommand)
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "--file",
    "path_file",
    type=click.Path(exists=True),
    help="File containing local paths (one per line)",
)
def gen_warmup(paths: tuple[str, ...], path_file: str | None):
    """Pre-register LOCAL repository paths in the analysis cache.

    Scans each local path and computes the real composite cache key
    so subsequent analysis runs get an instant cache hit.

    Note: Only local directory paths are supported. Remote URLs require
    manual cloning first — pass the path to the cloned repo.

    Examples:
        dspytools generate warmup /home/user/repos/numpy /home/user/repos/pandas
        dspytools generate warmup --file paths.txt
    """
    from dspytools.generate.cache import get_analysis_cache

    # No setup_dspy() — warmup is pure file scanning + hashing, no LM calls
    all_paths = list(paths)
    if path_file:
        with open(path_file) as f:
            all_paths.extend(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )

    if not all_paths:
        raise click.ClickException("no paths provided")

    # Validate paths exist
    from pathlib import Path

    invalid = [p for p in all_paths if not Path(p).exists() or not Path(p).is_dir()]
    if invalid:
        raise click.ClickException(
            f"Invalid or non-directory paths: {', '.join(invalid)}"
        )

    cache = get_analysis_cache()
    registered = cache.warmup(all_paths)
    ok(f"Warmed {len(registered)} path(s) in analysis cache")
    for path, key in registered.items():
        info(f"  {path} → {key}")
