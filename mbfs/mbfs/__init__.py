"""MBFS: MDP + Thompson Sampling feature selection."""

from .metrics import evaluate_selection
from .selectors import MBFSThompson

__version__ = "0.1.0"
__all__ = ["MBFSThompson", "evaluate_selection", "__version__"]