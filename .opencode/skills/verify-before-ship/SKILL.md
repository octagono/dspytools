---
name: verify-before-ship
description: Verify-before-ship gate — lint, test, import check before every commit
metadata:
  audience: developers
  workflow: git
---

## Verify-Before-Ship Gate

Before any commit, run this sequence:

### Step 1: Lint
```bash
ruff check --fix --unsafe-fixes
```
Must report zero errors.

### Step 2: Tests
```bash
pytest tests/ -q
```
Must pass 359/359, zero warnings.

### Step 3: Import Check
```bash
scripts/verify.sh
```

### Step 4: Quick Check
```bash
dspytools --help    # should show rich output
dspytools doctor --no-vllm --no-gpu --no-config  # all checks pass
```

### Commit
```bash
git add <files>
git commit -m "type: concise description"
```

Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `perf:`, `test:`
