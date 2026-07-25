"""Life-Harness layers — H1 Action, H2 Contract, H3 Trajectory, H4 Skills, H5 Self-Harness.

Full harness-so architecture implemented in dspytools:
  H1: Action Layer — API binding with invoke/bind/with_tools
  H2: Contract Layer — type contracts + runtime validation
  H3: Trajectory Layer — execution trace storage + replay + diff
  H4: Skills Layer — BM25 + embedding skill library (skills/)
  H5: Self-Harness Layer — SelfEvolveEngine + morphology + UCB (evolve/self_evolve.py)
"""

from dspytools.evolve.layers.action import Action, ActionLayer
from dspytools.evolve.layers.contract import (
    ContractLayer,
    ContractResult,
    ContractViolation,
)
from dspytools.evolve.layers.trajectory import TrajectoryLayer

__all__ = [
    "Action",
    "ActionLayer",
    "ContractLayer",
    "ContractResult",
    "ContractViolation",
    "TrajectoryLayer",
]
