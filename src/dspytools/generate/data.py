"""Ground truth training/dev data for llms.txt generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

# Example llms.txt for a well-known Python project
NUMPY_LLMS_TXT = """# NumPy

> Fundamental package for scientific computing with Python.

## Purpose
NumPy provides support for large multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

## Key Concepts
- **ndarray**: N-dimensional array object for efficient computation
- **Broadcasting**: Powerful mechanism for performing operations on arrays of different shapes
- **Vectorization**: Fast element-wise operations without explicit loops
- **Linear Algebra**: numpy.linalg module for matrix operations

## Architecture
- `numpy/core/`: Core C implementations and ndarray internals
- `numpy/lib/`: Utility functions and higher-level abstractions
- `numpy/linalg/`: Linear algebra operations

## Setup
```bash
pip install numpy
```"""

PANDAS_LLMS_TXT = """# Pandas

> Powerful data analysis and manipulation toolkit for Python.

## Purpose
Pandas provides high-performance, easy-to-use data structures and data analysis tools built on top of NumPy.

## Key Concepts
- **DataFrame**: 2-dimensional labeled data structure with columns of different types
- **Series**: 1-dimensional labeled array
- **GroupBy**: Split-apply-combine paradigm for data aggregation
- **Merge/Join**: SQL-like operations for combining datasets

## Architecture
- `pandas/core/`: Core DataFrame and Series implementations
- `pandas/io/`: File I/O (CSV, Excel, SQL, JSON, Parquet)
- `pandas/plotting/`: Matplotlib-integrated visualization

## Setup
```bash
pip install pandas
```"""

DSPY_LLMS_TXT = """# DSPy

> Programming—not Prompting—Foundation Models

## Purpose
DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used multiple times in a pipeline.

## Key Concepts
- **Signature**: Declarative I/O contracts for LM modules
- **Module**: Composable building blocks (ChainOfThought, ReAct, etc.)
- **Optimizer**: Automatic prompt/weight optimization (MIPROv2, GEPA)
- **Teleprompter**: DSPy's term for optimizers that tune prompts

## Architecture
- `dspy/predict/`: Core prediction modules (ChainOfThought, ReAct, CodeAct)
- `dspy/teleprompt/`: Optimizers (MIPROv2, GEPA, BootstrapFewShot, KNNFewShot)
- `dspy/adapters/`: Output formatting (Chat, JSON, XML, BAML)
- `dspy/primitives/`: Example, Prediction, Module, History

## Setup
```bash
pip install dspy-ai
```"""

SPACY_LLMS_TXT = """# spaCy

> Industrial-strength Natural Language Processing in Python.

## Purpose
spaCy provides fast, production-ready NLP with pre-trained pipelines and support for 75+ languages.

## Key Concepts
- **Doc**: Container for accessing linguistic annotations
- **Token**: Individual word with POS, dependency, lemma
- **Span**: Slice from a Doc
- **Pipeline**: Sequence of components (tagger, parser, NER, textcat)

## Architecture
- `spacy/lang/`: Language-specific data (tokenizers, lexemes)
- `spacy/pipeline/`: Trainable pipeline components
- `spacy/tokens/`: Doc, Span, Token data structures
- `spacy/matcher/`: Rule-based matching engines

## Setup
```bash
pip install spacy
python -m spacy download en_core_web_sm
```"""

TORCH_LLMS_TXT = """# PyTorch

> Tensors and Dynamic neural networks in Python with strong GPU acceleration.

## Purpose
PyTorch enables flexible deep learning research and production deployment with an imperative programming style.

## Key Concepts
- **Tensor**: Multi-dimensional array with GPU acceleration
- **Autograd**: Automatic differentiation engine
- **nn.Module**: Base class for all neural network layers
- **DataLoader**: Efficient batching, shuffling, multiprocessing

## Architecture
- `torch/nn/`: Neural network layers, loss functions, containers
- `torch/optim/`: Optimization algorithms (SGD, Adam, AdamW)
- `torch/autograd/`: Automatic differentiation
- `torch/utils/data/`: Dataset and DataLoader

## Setup
```bash
pip install torch
```"""


def build_ground_truth_examples() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build trainset/devset with real detailed llms.txt examples.

    Trainset: 5 repos (numpy, pandas, dspy, spacy, torch)
    Devset:   1 repo (fastapi) — unseen during training
    """

    trainset = [
        dspy.Example(
            repo_url="https://github.com/numpy/numpy",
            file_tree="numpy/core/\nnumpy/lib/\nnumpy/linalg/\nnumpy/array_api.py",
            readme_content="NumPy: fundamental package for scientific computing with Python.",
            package_files="=== pyproject.toml ===\n[project]\nname='numpy'\nversion='2.0.0'",
            llms_txt_content=NUMPY_LLMS_TXT,
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
        dspy.Example(
            repo_url="https://github.com/pandas-dev/pandas",
            file_tree="pandas/core/\npandas/io/\npandas/plotting/\npandas/__init__.py",
            readme_content="Pandas: powerful data analysis and manipulation toolkit.",
            package_files="=== pyproject.toml ===\n[project]\nname='pandas'\nversion='2.2.0'",
            llms_txt_content=PANDAS_LLMS_TXT,
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
        dspy.Example(
            repo_url="https://github.com/stanfordnlp/dspy",
            file_tree="dspy/predict/\ndspy/teleprompt/\ndspy/adapters/\ndspy/primitives/",
            readme_content="DSPy: Programming—not Prompting—Foundation Models",
            package_files="=== pyproject.toml ===\n[project]\nname='dspy-ai'\nversion='3.2.1'",
            llms_txt_content=DSPY_LLMS_TXT,
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
        dspy.Example(
            repo_url="https://github.com/explosion/spaCy",
            file_tree="spacy/lang/\nspacy/pipeline/\nspacy/tokens/\nspacy/matcher/",
            readme_content="spaCy: Industrial-strength NLP in Python.",
            package_files="=== setup.cfg ===\n[metadata]\nname=spacy\nversion=3.7.0",
            llms_txt_content=SPACY_LLMS_TXT,
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
        dspy.Example(
            repo_url="https://github.com/pytorch/pytorch",
            file_tree="torch/nn/\ntorch/optim/\ntorch/autograd/\ntorch/utils/data/",
            readme_content="PyTorch: Tensors and Dynamic neural networks with GPU acceleration.",
            package_files="=== setup.py ===\nsetup(name='torch', version='2.5.0')",
            llms_txt_content=TORCH_LLMS_TXT,
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
    ]

    devset = [
        dspy.Example(
            repo_url="https://github.com/tiangolo/fastapi",
            file_tree="fastapi/\nfastapi/routing.py\nfastapi/params.py\nfastapi/dependencies/",
            readme_content="FastAPI: high performance, easy to learn, fast to code, ready for production.",
            package_files="=== pyproject.toml ===\n[project]\nname='fastapi'\nversion='0.115.0'",
            llms_txt_content="""# FastAPI

> High-performance web framework for building APIs with Python.

## Purpose
FastAPI is a modern, fast web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Concepts
- **Path Parameters**: URL segments captured as typed parameters
- **Query Parameters**: Automatic validation from type hints
- **Dependency Injection**: Reusable dependencies with automatic caching
- **Pydantic Models**: Request/response schema and validation

## Architecture
- `fastapi/routing.py`: Route handling and HTTP method decorators
- `fastapi/params.py`: Parameter types (Path, Query, Body, Header)
- `fastapi/dependencies/`: Dependency injection system

## Setup
```bash
pip install fastapi uvicorn
```""",
        ).with_inputs("repo_url", "file_tree", "readme_content", "package_files"),
    ]

    return trainset, devset
