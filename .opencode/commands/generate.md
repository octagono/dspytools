---
description: Generate llms.txt for a repository
agent: build
---

Generate llms.txt documentation for a repository.

Usage: `/generate [url|path] [--local]`

Examples:
- Remote repo: `/generate https://github.com/numpy/numpy`
- Local repo: `/generate . --local`

Steps:
1. `scripts/dspytools generate llms-txt $ARGUMENTS`
2. Check the output quality with `scripts/dspytools generate batch`
3. Review the generated llms.txt file
