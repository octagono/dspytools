# Mojo Hybrid Acceleration

SIMD-accelerated hot paths for dspytools using Mojo 🔥 + Python.  
Follows a 95/5 hybrid architecture — keep Python for orchestration, drop to Mojo for CPU-bound inner loops.

## Directory Layout

Mojo source files live in `mojo_modules/` (named to avoid namespace collision
with the `mojo` pip package). Three `.mojo` files + `build.sh` + this README.

## Build Status — Mojo 1.0.0b2+

Mojo is available as a **pip-installable Python distribution** (`uv pip install mojo --prerelease allow`).
This distribution provides:

| Capability | Status |
|------------|--------|
| `mojo.importer` auto-compile hook | ✅ Works — `.mojo` files auto-compile on Python `import` |
| Python interop (`PythonObject`, `PythonModuleBuilder`) | ✅ Works with `from std.python` API |
| `ctypes` + `UnsafePointer` (manual pointer access) | ✅ Works — ctypes pointer access is reliable |
| `UnsafePointer.origin` inference (pip build) | 🔴 Broken — `Unsafepointer[mut, type](unsafe_from_address=addr)` can't infer `origin` |
| SIMD intrinsics (`simdwidthof`, `vectorize`) | 🔴 Requires full Mojo SDK (`modular install mojo`) |
| `mojo build --emit shared-lib` | 🔴 Requires full Mojo SDK |

### Practical impact

dspytools uses Mojo via the **`mojo.importer` auto-compile hook** — no separate build step.
All three bridges do:

```python
import mojo.importer       # activates .mojo → .so auto-compile
import sprt                # auto-compiles sprt.mojo from sys.path
```

The `mojo build` step (`build.sh`) is only needed when the full Mojo SDK is available and SIMD
acceleration or pre-built `.so` files are desired. Without the full SDK, the bridges use
**ctypes pointer access** for array operations instead of SIMD vectorized loads, and pure Python
fallbacks for functions that require full SIMD support.

### Key Mojo 1.0.0b2 API differences

| Concept | Old (0.x docs) | New (1.0.0b2) |
|---------|---------------|----------------|
| Python interop import | `from python import PythonObject` | `from std.python import PythonObject` |
| Module builder import | `from python.bindings import PythonModuleBuilder` | `from std.python.bindings import PythonModuleBuilder` |
| Function declaration | `fn` | `def` (or `fn` still works) |
| Variable declaration | `let` | `var` (or `let` still works) |
| Python export decorator | `@export fn PyInit_*()` | `@export def PyInit_*() abi("C")` |
| Compile-time alias | `alias` | `comptime` |
| Max Python params | Unlimited | **8** `PythonObject` params max |
| Keyword format args | `String.format(key=val)` | Broken — use concatenation |

## Architecture

```
mojo_modules/
├── build.sh           # Compile all modules → .so (needs full SDK)
├── README.md          # This file
├── vector_utils.mojo  # Phase 1 — Float32 array ↔ bytes serialization
├── sprt.mojo          # Phase 2 — Sequential Probability Ratio Test
└── bm25.mojo          # Phase 3 — BM25 information retrieval scoring

src/dspytools/
├── graph/
│   ├── cache_mojo_bridge.py  # Phase 1 bridge (→ load_store_float32)
│   ├── cache.py              # Integration: _vec_to_blob → bridge
│   └── benchmark.py          # Fuzz + throughput benchmark
├── core/
│   └── sprt_mojo_bridge.py   # Phase 2 bridge (→ sprt_evaluate)
└── skills/
    └── bm25_mojo_bridge.py   # Phase 3 bridge (→ bm25_score_docs)
```

## Design Pattern

Each phase follows a three-layer pattern:

```
Python caller (existing code)
    ↓ function call
Python bridge (cache_mojo_bridge.py / sprt_mojo_bridge.py / bm25_mojo_bridge.py)
    ├── HAS_MOJO guard → mojo.importer auto-compile → Mojo function call
    └── !HAS_MOJO guard → pure Python fallback (identical semantics)
```

Each bridge has:
- **`HAS_MOJO: bool`** — set at import time, True if Mojo module loaded
- **`has_mojo() -> bool`** — runtime check if Mojo is active
- **Pure Python fallback** — runs when Mojo module fails to load

## Quick Start

```python
# Mojo auto-compiles on import — just run:
python3 -c "
import mojo.importer
import vector_utils
import numpy as np
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
blob = vector_utils.load_store_float32(arr)
print('Mojo vector_utils works!', len(blob), 'bytes')
"
```

## Phase 1 — Vector Serialization

**File:** `mojo_modules/vector_utils.mojo`  
**Bridge:** `src/dspytools/graph/cache_mojo_bridge.py`  
**Integration:** `SemanticCache._vec_to_blob()` in `graph/cache.py`

### Implementation

Uses ctypes pointer scan + manual byte copy since `UnsafePointer` origin inference
is broken in the pip build. SIMD `vectorize[load, store]` path requires full Mojo SDK.

### Verified

✅ `load_store_float32` compiles and runs end-to-end via `mojo.importer`

### Fallback

```python
np.asarray(vec, dtype=np.float32).tobytes()
```

## Phase 2 — SPRT Math

**File:** `mojo_modules/sprt.mojo`  
**Bridge:** `src/dspytools/core/sprt_mojo_bridge.py`  
**Integration:** `SelfEvolveEngine.validate_and_deploy()` in `evolve/self_evolve.py`

### Implementation

Uses `ctypes` for array pointer access and `Python.evaluate("dict")` for result
construction (avoids `String.format` which lacks keyword arg support in 1.0.0b2).

### Fallback

Pure Python loop with per-element log-likelihood ratio accumulation and decision boundary checks.

## Phase 3 — BM25 Scoring

**File:** `mojo_modules/bm25.mojo`  
**Bridge:** `src/dspytools/skills/bm25_mojo_bridge.py`  
**Integration:** `SkillLoader.search()` in `skills/loader.py`

### Implementation note

`def_function` has an **8-parameter limit** for `PythonObject` arguments. BM25's original
9-param signature was refactored to pack document count and term count into a `(n_docs, n_terms)`
tuple passed as a single parameter.

### Fallback

Pure Python double-nested loop with per-document, per-term BM25 computation.

## Benchmark

```bash
python -m dspytools.graph.benchmark --all
```

Tests correctness (fuzz: 500 random vectors) and throughput (all sizes 1–4096).

## Build (full SDK only)

```bash
# Build all modules
./mojo_modules/build.sh

# Build specific modules
./mojo_modules/build.sh vector_utils
./mojo_modules/build.sh sprt bm25
```

Prerequisites: Full Mojo SDK installed via `modular install mojo`.

## When to Add a New Phase

1. Identify a CPU-bound inner loop in Python (profiler: py-spy, cProfile)
2. The loop is data-parallel (SIMD-friendly) with no Python FFI calls per iteration
3. Data is already contiguous in memory (numpy arrays, flat lists)
4. Write `.mojo` module exposing a single `fn` via `PythonModuleBuilder`
5. Write Python bridge with `HAS_MOJO` guard and pure Python fallback
6. Wire the bridge into existing code behind a simple function call
7. Add benchmark/verification harness
8. Update this README
