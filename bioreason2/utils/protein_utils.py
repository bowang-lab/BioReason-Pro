"""
Protein sequence utilities for BioReason2.
"""

from typing import Union, List
import numpy as np
import torch

# Type definition for protein inputs
ProteinInput = Union[
    str,
    List[int],
    np.ndarray,
    torch.Tensor,
    List[str],
    List[List[int]],
    List[np.ndarray],
    List[torch.Tensor],
]
