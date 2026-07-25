---
description: Run lint + tests + import verification gate
agent: build
---

Run the full verification gate before shipping changes.

1. Run: `scripts/verify.sh`
2. If ruff fails: `.venv/bin/ruff check --fix --unsafe-fixes`
3. If tests fail, identify the root cause and fix it
4. Check git status and ensure no unintended changes
5. Re-run `scripts/verify.sh` until all checks pass
6. Verify `dspytools --help` still shows clean output

Only proceed with commit after all checks pass.
