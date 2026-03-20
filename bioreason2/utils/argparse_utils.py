"""
Argument parsing utilities for BioReason2.
"""

import argparse
from typing import Union


def str2bool(v: Union[str, bool]) -> bool:
    """
    Helper function to properly parse boolean arguments from command line.

    This function addresses the common issue where argparse's type=bool doesn't
    work as expected (any non-empty string becomes True).

    Args:
        v: String or boolean value to convert

    Returns:
        Boolean value

    Raises:
        argparse.ArgumentTypeError: If the input cannot be converted to boolean

    Examples:
        >>> str2bool('True')
        True
        >>> str2bool('false')
        False
        >>> str2bool('1')
        True
        >>> str2bool('0')
        False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")
