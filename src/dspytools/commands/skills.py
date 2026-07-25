"""dspytools skills — BM25-indexed skill library with auto-generation."""

from __future__ import annotations

from dspytools.cli.output import console, info, ok, panel, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click


@click.group(name="skills", cls=LLMGroup)
def skills_cmd():
    """Manage reusable DSPy skills (generate, compile, search)."""


@skills_cmd.command(name="list", cls=LLMCommand)
def skills_list():
    """List all available skills."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    skills = mgr.list_skills()

    if skills:
        rows = [
            [s.name, s.description[:60], "✓" if s.has_program else "✗"] for s in skills
        ]
        table("Skills", ["Name", "Description", "Compiled"], rows)
    else:
        info("No skills found. Create with: dspytools skills create")


@skills_cmd.command(name="create", cls=LLMCommand)
@click.argument("name")
@click.argument("description")
@click.option("--signature", "-s", default="question -> answer", help="DSPy signature")
@click.option("--body", help="Markdown body for SKILL.md")
def skills_create(name: str, description: str, signature: str, body: str | None):
    """Create a new skill with SKILL.md."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    skill = mgr.create_skill(name, description, signature, body or "")
    ok(f"Skill '{name}' created")
    info(f"Location: {skill.path}")


@skills_cmd.command(name="search", cls=LLMCommand)
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of results")
def skills_search(query: str, k: int):
    """BM25 search for skills matching a query."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    results = mgr.search(query, k)

    if results:
        rows = [
            [s.name, s.description[:60], "✓" if s.has_program else "✗"] for s in results
        ]
        table(f"BM25 Search: '{query}'", ["Name", "Description", "Compiled"], rows)
    else:
        info(f"No skills match '{query}'")


@skills_cmd.command(name="compile", cls=LLMCommand)
@click.argument("name")
@click.option(
    "--optimizer",
    "-o",
    default="labeled_few_shot",
    type=click.Choice(["labeled_few_shot", "bootstrap_few_shot"]),
)
def skills_compile(name: str, optimizer: str):
    """Compile a skill's DSPy program."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    result = mgr.compile_skill(name, optimizer=optimizer)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        ok(f"Skill '{name}' compiled with {result['optimizer']}")


@skills_cmd.command(name="generate-from-run", cls=LLMCommand)
@click.argument("run_id")
@click.argument("skill_name")
@click.argument("description")
def skills_generate_from_run(run_id: str, skill_name: str, description: str):
    """Generate a skill from an existing compiled program."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    skill = mgr.generate_from_program(run_id, skill_name, description)

    if skill:
        ok(f"Skill '{skill_name}' generated from run {run_id}")
    else:
        console.print(f"[red]Run '{run_id}' not found or has no compiled program[/red]")


@skills_cmd.command(name="auto-optimize", cls=LLMCommand)
@click.argument("name")
@click.option("--trainset", "-t", help="Path to JSON training examples")
def skills_auto_optimize(name: str, trainset: str | None):
    """Auto-optimize a skill using the GFL pipeline."""
    from dspytools.core.setup import setup_dspy
    from dspytools.skills import SkillManager

    setup_dspy()

    # Load trainset if provided
    examples = None
    if trainset:
        from dspytools.core.loaders import load_trainset

        examples = load_trainset(trainset)

    mgr = SkillManager()
    result = mgr.auto_optimize_skill(name, trainset=examples)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        panel(
            "Skill Optimized",
            f"[bold]Skill:[/] {result['skill']}\n"
            f"[bold]Best:[/] {result['best']}\n"
            f"[bold]Score:[/] {result['score']:.2f}",
            border_style="green",
        )


@skills_cmd.command(name="show", cls=LLMCommand)
@click.argument("name")
def skills_show(name: str):
    """Show a skill's details."""
    from dspytools.skills import SkillManager

    mgr = SkillManager()
    skill = mgr.search(name, k=1)
    if skill and skill[0].name == name:
        md_path = skill[0].path / "SKILL.md" if skill[0].path else None
        if md_path and md_path.exists():
            console.print(md_path.read_text())
        else:
            console.print(f"[red]Skill '{name}' found but SKILL.md missing[/red]")
    else:
        console.print(f"[red]Skill '{name}' not found[/red]")


@skills_cmd.command(name="find", cls=LLMCommand)
@click.argument("query")
@click.option("--k", default=10, type=int, help="Number of results")
@click.option("--category", "-c", help="Filter by category")
def skills_find(query: str, k: int, category: str | None):
    """Search for skills in the open agent skills ecosystem (skills.sh).

    \b
    Examples:
        dspytools skills find react
        dspytools skills find testing --category testing
        dspytools skills find dspy --k 5
    """
    from dspytools.skills.discovery import search_external, try_skills_sh_api

    live = try_skills_sh_api(query)
    if live:
        results = []
        for item in live[:k]:
            from dspytools.skills.discovery import ExternalSkill

            results.append(
                ExternalSkill(
                    name=item.get("name", "?"),
                    source=item.get("source", "?"),
                    description=item.get("description", ""),
                    installs=item.get("installs", 0),
                )
            )
        info("Results from skills.sh live API")
    else:
        results = search_external(query, k=k)
        if category:
            results = [r for r in results if r.category == category]
            results = results[:k]
        info("Results from curated dataset")

    if results:
        rows = [
            [
                s.name,
                s.source,
                f"{s.installs / 1000:.0f}K",
                s.category,
                s.description[:50],
            ]
            for s in results
        ]
        table(
            f"Skills matching '{query}'",
            ["Name", "Source", "Installs", "Category", "Description"],
            rows,
        )

        top = results[0]
        console.print(f"\n[bold]Top match:[/] {top.name}")
        console.print(f"  {top.description}")
        console.print(
            f"  [dim]Installs: {top.installs / 1000:.0f}K  |  Source: {top.source}[/dim]"
        )
        console.print(f"\n  [green]Install:[/] npx skills add {top.source}@{top.name}")
    else:
        info(f"No skills match '{query}'. Try: dspytools skills discover")


@skills_cmd.command(name="discover", cls=LLMCommand)
@click.option("--category", "-c", help="Filter by category")
@click.option("--k", default=20, type=int, help="Number of results")
def skills_discover(category: str | None, k: int):
    """Browse popular skills from the open ecosystem.

    \b
    Examples:
        dspytools skills discover
        dspytools skills discover --category ai-ml
        dspytools skills discover --category testing --k 5
    """
    from dspytools.skills.discovery import list_categories, popular_skills

    if category is None:
        cats = list_categories()
        console.print("\n[bold]Available categories:[/]")
        for cat in cats:
            count = len(popular_skills(100, category=cat))
            console.print(f"  - {cat} ({count} skills)")

    results = popular_skills(k=k * 2 if category else k, category=category)

    if results:
        rows = [
            [
                s.name,
                s.source,
                f"{s.installs / 1000:.0f}K",
                s.category,
                s.description[:60],
            ]
            for s in results
        ]
        title = "Popular Skills" + (f" ({category})" if category else "")
        table(title, ["Name", "Source", "Installs", "Category", "Description"], rows)

        console.print("\n[bold]Install a skill:[/]")
        for s in results[:3]:
            cmd = f"npx skills add {s.source}@{s.name}"
            console.print(
                f"  [green]{cmd}[/] [dim]({s.installs / 1000:.0f}K installs)[/dim]"
            )
    else:
        info("No skills found in this category.")


@skills_cmd.command(name="categories", cls=LLMCommand)
def skills_categories():
    """List all skill categories in the ecosystem."""
    from dspytools.skills.discovery import list_categories, popular_skills

    cats = list_categories()
    rows = [
        [
            cat,
            str(len(popular_skills(100, category=cat))),
            popular_skills(3, category=cat)[0].name
            if popular_skills(1, category=cat)
            else "-",
        ]
        for cat in cats
    ]
    table("Skill Categories", ["Category", "Count", "Top Skill"], rows)
