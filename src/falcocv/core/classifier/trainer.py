"""Projection head trainer — delegates to pure-numpy implementation.

Replaced the torch-based implementation with an equivalent numpy version
to eliminate the torch runtime dependency.  Verified bit-identical gradient
computation in tests/test_trainer_equivalence.py.
"""

from .trainer_numpy import train_projection_head

__all__ = ["train_projection_head"]
