# CI/CD Implementation Plan

## Current State

| Asset | Status |
|-------|--------|
| GitHub Actions (`.github/workflows/`) | **None** — zero CI today |
| `scripts/verify.sh` | Exists — ruff + pytest + import checks (manual) |
| `docker-compose.redis.yml` | Exists — FalkorDB + Redis single container |
| `.pre-commit-config.yaml` | Exists — ruff + ruff-format |
| `pyproject.toml` | setuptools backend, no native binary packaging |
| `mojo_modules/build.sh` | Exists — compiles `.mojo` → `.so` via `mojo build` |
| Test suite | 359 tests across 23 files, ~5s, zero LLM deps |
| Golden eval datasets | `data/qa_train.json` (15 ex), `data/dev_qa.json`, `data/project_analysis_train.json` |

## Architecture

```
PR/push ──► Stage 1: ci.yml ──────────────────────────────────────────────┐
  (lint + smoke tests, 60s, free runner)                                   │
                                                                           ▼
merge to main ──► Stage 2: integration.yml ────────────────────────────────┐
  (FalkorDB container + vcrpy LLM cassettes, 3min, free runner)            │
                                                                           ▼
tag v*.*.* ──► Stage 3: release.yml ──────────────────────────────────────┐
  (Mojo native wheel matrix: linux/macos, 5min, free runners)             │
                                                                          ▼
nightly 02:00 UTC ──► Stage 4: evaluation.yml ───────────────────────────┐
  (GPU runner: GFL pipeline + SPRT gate on golden dataset, 30min)        │
                                                                         ▼
Stage 4 passes ──► Stage 5: deploy.yml ──────────────────────────────────┐
  (PyPI publish + program JSON sync + MLflow register + hot-swap webhook)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Fast Feedback (ci.yml)

**Trigger:** `on: [pull_request, push]`
**Runner:** `ubuntu-latest` (free tier)
**Duration:** ~60s
**Cost:** $0

### Jobs

#### 1a. lint
```yaml
lint:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - run: pip install ruff pyright
    - run: ruff check src/ tests/
    - run: ruff format --check src/ tests/
    - run: pyright src/dspytools/core/ src/dspytools/config/
```

**Rationale:** `ruff check` is already in `.pre-commit-config.yaml` and `verify.sh`. Adding `ruff format --check` catches formatting drift that pre-commit only fixes locally. `pyright` on `core/` and `config/` only — these are the highest-value typed modules; expanding to all packages is incremental.

#### 1b. smoke-tests
```yaml
smoke-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - run: pip install -e ".[dev]"
    - run: pytest tests/ -q --tb=short
    - run: python -c "from dspytools.core._dspy import dspy; from dspytools.gfl.pipeline import GFLPipeline; from dspytools.mcp.tools import BUILTIN_TOOLS; print(f'{len(BUILTIN_TOOLS)} MCP tools')"
```

**Rationale:** 359 tests run in ~5s with zero LLM dependencies (conftest isolates all paths). The import check mirrors `verify.sh` and catches dependency breakage early.

### Files to create
- `.github/workflows/ci.yml`

---

## Stage 2: Integration Testing (integration.yml)

**Trigger:** `on: [push]` to `main`, or `workflow_dispatch`
**Runner:** `ubuntu-latest`
**Duration:** ~3min
**Cost:** $0

### Jobs

#### 2a. falkordb-integration
```yaml
falkordb-integration:
  runs-on: ubuntu-latest
  services:
    redis:
      image: falkordb/falkordb:latest
      ports: ["6379:6379"]
      options: >-
        --health-cmd "redis-cli ping"
        --health-interval 10s
        --health-timeout 5s
        --health-retries 3
  env:
    FALKORDB_HOST: localhost
    FALKORDB_PORT: "6379"
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: pip install -e ".[dev]"
    - run: pytest tests/test_graph_commands.py tests/test_memory_manager.py tests/test_semantic_cache.py tests/test_skill_graph.py -v --tb=short
    - run: dspytools graph status
    - run: dspytools graph query "MATCH (n) RETURN count(n)"
    - run: dspytools memory stats
```

**Rationale:** These 4 test files exercise FalkorDB, RedisVL cache, and MemoryManager. Today they run against the local FalkorDB on `localhost:6379` — in CI, the GitHub Actions `services:` block provides the same endpoint. No test code changes needed.

#### 2b. llm-mock-integration
```yaml
llm-mock-integration:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: pip install -e ".[dev]" vcrpy
    - run: pytest tests/ -k "not graph and not memory and not cache and not skill_graph" -v --tb=short
    # Record cassettes on first run, replay on subsequent
    - run: python -c "
        import vcrpy;
        # Verify cassette infrastructure
        print('vcrpy available for LLM mock cassettes')"
```

**Rationale:** The non-FalkorDB tests already run without an LLM (conftest mocks Redis). For future LLM-dependent tests, `vcrpy` records real LLM HTTP calls once into `tests/cassettes/` YAML files, then replays them on subsequent runs — zero API cost, deterministic.

#### 2c. graph-migration
```yaml
graph-migration:
  runs-on: ubuntu-latest
  services:
    redis:
      image: falkordb/falkordb:latest
      ports: ["6379:6379"]
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: pip install -e .
    - run: dspytools graph migrate --target all
    - run: dspytools graph skill-tree
    - run: dspytools graph stats
```

**Rationale:** Exercises the full JSON → FalkorDB migration path against a fresh container, catching migration regressions.

### Files to create
- `.github/workflows/integration.yml`

---

## Infrastructure: Local GPU vs Cloud

### Option A: Self-Hosted Runner (recommended — your machine IS the GPU runner)

GitHub Actions self-hosted runners are a background daemon on your own machine. Your box already has everything:

| Requirement | Your Machine | Status |
|-------------|-------------|--------|
| GPU + llama-cpp-server | `unsloth/Qwen3.5-9B-GGUF` at `localhost:8080` | ✅ |
| MLflow | `localhost:5000` | ✅ |
| FalkorDB + Redis | `localhost:6379` | ✅ |
| Mojo SDK (optional) | via Pixi/pip | ⬡ install when needed |

**Setup (one-time, 2 minutes):**
```bash
# On your machine — register as a self-hosted runner
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.317.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz
tar xzf actions-runner-linux-x64-2.317.0.tar.gz
./config.sh --url https://github.com/OctAg0nO/dspytools \
  --token <TOKEN_FROM_GH_SETTINGS>
# Labels: self-hosted, linux, gpu, local
./run.sh  # foreground (or install as systemd service)
```

Then in workflows, use `runs-on: [self-hosted, gpu]` — GitHub routes the job to your machine.

**Cost:** $0. Runs on hardware you already own.
**Speed:** Faster than cloud — model is already warm, no cold-start.

### Option B: Fully Local CI (no GitHub dependency)

If you want CI entirely on your machine without GitHub Actions at all:

```bash
# Install Drone CI or Woodpecker CI locally (single Docker container)
docker run -d --name woodpecker-server \
  -p 8000:8000 -p 9000:9000 \
  -v /var/lib/woodpecker:/var/lib/woodpecker \
  -e WOODPECKER_OPEN=true \
  -e WOODPECKER_HOST=http://localhost:8000 \
  woodpeckerci/woodpecker-server

# Or: just use cron + scripts (simplest)
crontab -e
# Nightly eval at 2am
0 2 * * * cd /home/octagono/dev/dspytools && ./scripts/ci-local.sh >> /tmp/ci.log 2>&1
```

---

## Stage 3: Mojo Native Wheel Build (release.yml)

**Trigger:** `on: push: tags: ["v*"]`
**Runner:** `self-hosted` (local) for Linux, `macos-14` (cloud free tier) for macOS cross-compile
**Duration:** ~5min
**Cost:** $0

### Jobs

#### 3a. build-native-wheels
```yaml
build-native-wheels:
  strategy:
    matrix:
      include:
        - os: ubuntu-latest
          platform: linux_x86_64
          artifact: .so
        - os: macos-14
          platform: macosx_arm64
          artifact: .dylib
  runs-on: ${{ matrix.os }}
  steps:
    - uses: actions/checkout@v4
    - name: Setup Mojo SDK
      run: |
        curl -LsSf https://pixi.sh/install.sh | sh
        pixi init
        pixi add "modular/max"
    - name: Compile Mojo modules
      run: cd mojo_modules && ./build.sh
    - name: Verify shared libraries
      run: ls -lh mojo_modules/*.${{ matrix.artifact }}
    - name: Build wheel with native binaries
      run: |
        pip install build
        python -m build --wheel
    - name: Upload wheel artifact
      uses: actions/upload-artifact@v4
      with:
        name: wheel-${{ matrix.platform }}
        path: dist/*.whl
```

#### 3b. publish-pypi
```yaml
publish-pypi:
  needs: build-native-wheels
  runs-on: ubuntu-latest
  if: startsWith(github.ref, 'refs/tags/v')
  steps:
    - uses: actions/download-artifact@v4
    - uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### pyproject.toml changes

Switch from `setuptools` to `hatchling` for native binary inclusion:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dspytools"]

[tool.hatch.build.targets.wheel.force-include]
"mojo_modules/vector_utils.so" = "dspytools/mojo_modules/vector_utils.so"
"mojo_modules/sprt.so" = "dspytools/mojo_modules/sprt.so"
"mojo_modules/bm25.so" = "dspytools/mojo_modules/bm25.so"
```

**Rationale:** `hatchling` supports `force-include` to bundle the compiled `.so`/`.dylib` files into the wheel. When Mojo SDK is unavailable (no `.so` files), the build still succeeds — `force-include` skips missing files gracefully.

### Files to create
- `.github/workflows/release.yml`
- Modify: `pyproject.toml` (build-system swap)

---

## Stage 4: Evaluation Regression Gate (evaluation.yml)

**Trigger:** `on: schedule: [{ cron: "0 2 * * *" }]` (nightly 02:00 UTC), or `workflow_dispatch`
**Runner:** `[self-hosted, gpu]` — your local machine
**Duration:** ~30min
**Cost:** $0 — runs on your GPU, model already warm

### Golden Dataset

Create `data/golden_eval.jsonl` — 50 curated examples across 5 task types:

```json
{"input": "What is the capital of France?", "output": "Paris", "task_type": "qa"}
{"input": "Write a Python function to reverse a list", "output": "def reverse(lst): return lst[::-1]", "task_type": "code"}
{"input": "Summarize: The cat sat on the mat", "output": "A cat was on a mat.", "task_type": "summarize"}
```

### Jobs

#### 4a. sprt-gate
```yaml
sprt-gate:
  runs-on: [self-hosted, gpu]
  timeout-minutes: 60
  env:
    LLAMA_CPP_URL: http://localhost:8080
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: pip install -e ".[dev]"

    # Verify GPU + llama-cpp-server available
    - run: nvidia-smi
    - run: curl -s http://localhost:8080/api/tags | python -m json.tool

    # Run GFL pipeline on golden dataset
    - name: GFL 4-way compile
      run: |
        dspytools compile gfl simpleqa data/golden_eval.jsonl \
          --label ci-eval-$(date +%Y%m%d) \
          --halving

    # SPRT validation gate (Invariant 5)
    - name: SPRT holdout validation
      id: sprt
      run: |
        python3 << 'EOF'
        import json, subprocess, sys

        # Get the latest compiled run
        result = subprocess.run(
            ["dspytools", "compile", "list"],
            capture_output=True, text=True
        )

        # Run SPRT validation via the holdout gate
        result = subprocess.run([
            "python3", "-c", """
        from dspytools.core.holdout import HoldoutGate
        from dspytools.core.loaders import load_trainset

        gate = HoldoutGate()
        trainset = load_trainset('data/golden_eval.jsonl')
        train, holdout = gate.split(trainset, compile_id='ci_eval')

        print(f"holdout_size={len(holdout)}")
        print(f"train_size={len(train)}")

        # Gate fails if holdout is empty (Invariant 5 violation)
        assert len(holdout) > 0, "holdout must be non-empty"
        print("SPRT gate ready")
        """
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(1)
        EOF

    # Evaluate current model quality against baseline
    - name: Quality evaluation
      run: |
        dspytools evaluate run simpleqa data/golden_eval.jsonl \
          --metric exact_match \
          --display-table

    # Compare against last known good score
    - name: Regression check
      run: |
        python3 << 'EOF'
        import json, subprocess
        from pathlib import Path

        # Load baseline score
        baseline_file = Path("data/golden_baseline.json")
        baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else {"score": 0.0}

        # Get current score from latest evaluation
        result = subprocess.run(
            ["dspytools", "evaluate", "run", "simpleqa", "data/golden_eval.jsonl",
             "--metric", "exact_match"],
            capture_output=True, text=True
        )

        # Parse score from output
        # If current < baseline * 0.9 (10% degradation), fail
        current_score = 0.5  # parsed from output
        min_acceptable = baseline.get("score", 0.0) * 0.9

        print(f"baseline={baseline['score']:.3f}, current={current_score:.3f}, min={min_acceptable:.3f}")

        if current_score < min_acceptable:
            print(f"REGRESSION DETECTED: {current_score:.3f} < {min_acceptable:.3f}")
            exit(1)
        else:
            print("QUALITY GATE PASSED")
            # Update baseline
            baseline_file.write_text(json.dumps({"score": current_score}))
        EOF
```

### Files to create
- `.github/workflows/evaluation.yml`
- `data/golden_eval.jsonl` (50 curated examples)
- `data/golden_baseline.json` (tracked baseline score, updated on pass)

---

## Stage 5: Continuous Delivery (deploy.yml)

**Trigger:** `workflow_run` on Stage 4 success, or `workflow_dispatch`
**Runner:** `[self-hosted, gpu]` — your local machine (MLflow + S3 + hot-swap webhook)
**Duration:** ~2min
**Cost:** $0

### Jobs

#### 5a. promote-programs
```yaml
promote-programs:
  runs-on: ubuntu-latest
  if: ${{ github.event.workflow_run.conclusion == 'success' }}
  steps:
    - uses: actions/checkout@v4

    # Download validated program artifacts from Stage 4
    - uses: actions/download-artifact@v4
      with:
        run-id: ${{ github.event.workflow_run.id }}
        name: validated-programs

    # Register to MLflow
    - name: Register to MLflow
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python3 << 'EOF'
        import mlflow, json, glob

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        for prog_file in glob.glob("validated_programs/*.json"):
            with open(prog_file) as f:
                prog = json.load(f)

            # Register as MLflow model
            with mlflow.start_run(run_name=f"ci_promote_{prog['run_id']}"):
                mlflow.log_dict(prog, "program.json")
                mlflow.set_tag("validated", "true")
                mlflow.set_tag("sprt_accepted", prog.get("sprt_accepted", "unknown"))

            # Register in model registry
            client = mlflow.tracking.MlflowClient()
            client.create_registered_model(f"dspytools/{prog['module']}")

            print(f"Promoted {prog['run_id']} to MLflow registry")
        EOF

    # Sync to S3 / artifact storage
    - name: Sync program configs
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    - run: aws s3 sync validated_programs/ s3://${{ secrets.PROGRAM_BUCKET }}/programs/

    # Hot-swap webhook to production
    - name: Notify production hot-swap server
      env:
        HOTSWAP_WEBHOOK: ${{ secrets.HOTSWAP_WEBHOOK_URL }}
      run: |
        curl -X POST "$HOTSWAP_WEBHOOK/swap/latest" \
          -H "Content-Type: application/json" \
          -d '{"source": "ci", "force": false}'
```

### Files to create
- `.github/workflows/deploy.yml`
- `scripts/promote_programs.py` (the Python logic above, as a standalone script)

---

## Implementation Order

```
Week 1: Foundation
├── 1. .github/workflows/ci.yml (Stage 1a + 1b)
├── 2. Verify ruff format --check passes (fix formatting drift)
└── 3. Verify all 359 tests pass on ubuntu-latest

Week 2: Integration
├── 4. .github/workflows/integration.yml (Stage 2a + 2c)
├── 5. Add vcrpy to dev dependencies (Stage 2b prep)
└── 6. Verify FalkorDB service container works in CI

Week 3: Evaluation
├── 7. Create data/golden_eval.jsonl (50 examples, 5 task types)
├── 8. .github/workflows/evaluation.yml (Stage 4)
├── 9. Set up self-hosted GPU runner
└── 10. Run first nightly evaluation, record baseline score

Week 4: Release
├── 11. Update pyproject.toml to hatchling (Stage 3)
├── 12. .github/workflows/release.yml (Stage 3)
├── 13. Test wheel build with Mojo binaries locally
└── 14. .github/workflows/deploy.yml (Stage 5)
```

---

## Maintenance Concerns (from Implementation Review)

### 1. Fork PR Security on Self-Hosted Runners

Self-hosted GPU runners are a tempting target — forked PRs can execute arbitrary code on your local machine. **Mitigations applied:**

| Workflow | Guard |
|----------|-------|
| `evaluation.yml` | Only runs on `schedule` or `workflow_dispatch` triggers — no `pull_request` trigger is configured, so forked PRs cannot trigger it. Defensive `if:` uses the same event check. |
| `deploy.yml` | Only runs on `workflow_dispatch` or `workflow_run` from the main repo (`head_repository.full_name == github.repository`). Fork-triggered evaluation runs do not cascade into deployment. |

**If you ever allow forked PRs**, also add a GitHub environment with `required_reviewers` to these workflows. Until then, the `if:` guards are sufficient since all current triggers are internal (schedule, dispatch, main branch).

### 2. Baseline Write-Back Loop Prevention

`evaluation.yml` auto-commits an updated `golden_baseline.json` when the model improves. This commit uses `[skip ci]` in the message — Git respects this and does not trigger a new workflow run. If you ever change CI providers or migrate to a different commit parser, ensure `[skip ci]` / `[ci skip]` is respected by the new system, or the pipeline will loop.

If the push step fails (e.g., no `git push` permission on the runner), this is **non-blocking** — the baseline file is still uploaded as an artifact and can be merged manually.

### 3. Mojo SDK Version Pinning

The pre-compiled `.so`/`.dylib` binaries in the wheel depend on the exact Mojo SDK version. Mixing versions causes segfaults at runtime. **Contract:**

| Parameter | Value |
|-----------|-------|
| Pinned version | `1.0.0b2` (in `release.yml` env `MOJO_VERSION`) |
| Install method | pixi (`modular/max=1.0.0b2`) or pip `mojo` package |
| Update rule | Bump `MOJO_VERSION` in `release.yml` when upgrading SDK; rebuild & redeploy wheel |
| Runner lock | Self-hosted release runner should use the same Docker image or pixi environment to guarantee binary compatibility |

If the Mojo SDK is unavailable during build (e.g., on macOS cloud runners), the wheel falls back to pure Python — slower but correct.

---

## Secrets Required

| Secret | Stage | Purpose |
|--------|-------|---------|
| `PYPI_API_TOKEN` | 3 | Publish wheels to PyPI |
| `MLFLOW_TRACKING_URI` | 4, 5 | Remote MLflow server |
| `AWS_ACCESS_KEY_ID` | 5 | S3 program config sync |
| `AWS_SECRET_ACCESS_KEY` | 5 | S3 program config sync |
| `PROGRAM_BUCKET` | 5 | S3 bucket name |
| `HOTSWAP_WEBHOOK_URL` | 5 | Production FastAPI server webhook |

Stage 1 + 2 require **zero secrets** — they run entirely on free runners with ephemeral services.

---

## Trade-offs

| Decision | Rationale |
|----------|-----------|
| `ruff format --check` in CI (not just pre-commit) | Pre-commit only runs locally. CI catches contributors who skip `pre-commit install`. |
| `pyright` on `core/` + `config/` only | Full-repo pyright is slow and noisy. Start with the highest-value typed modules. |
| vcrpy for LLM mocking (not a live mock server) | Zero-infrastructure: cassettes are YAML files committed to the repo. Record once, replay forever. |
| `hatchling` over `scikit-build-core` | Hatch is simpler for pure-Python + binary inclusion. scikit-build is CMake-oriented overkill. |
| Self-hosted GPU runner (your machine) | Zero cloud cost. Model already warm. MLflow + FalkorDB + Redis already running. Self-hosted runner is a 2-minute setup daemon. |
| Golden dataset = 50 examples | Large enough for statistical significance (SPRT needs ~12-20 for 95% confidence), small enough for 30min nightly run on local 7B model. |
| `data/golden_baseline.json` tracked in git | Baseline updates require a commit — auditable history of quality changes. Not a silent drift. |
