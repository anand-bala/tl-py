"""
Efficient robust monitoring for STL.

[1] A. Donzé, T. Ferrère, and O. Maler, “Efficient Robust Monitoring for STL,”
    in Computer Aided Verification, 2013, pp. 264–279.
"""

import numpy as np
import pandas as pd

import temporal_logic.signal_tl as stl

BOTTOM = -np.inf
TOP = np.inf


def _compute_not(y: pd.Series) -> pd.Series:
    return -1 * y


def _compute_or(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.max(axis=1)


def _compute_or_binary(x: pd.Series, y: pd.Series):
    return pd.DataFrame([x, y]).max(axis=1)


def _compute_and(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.min(axis=1)


def _compute_and_binary(x: pd.Series, y: pd.Series):
    return pd.DataFrame([x, y]).min(axis=1)


def efficient_robustness(phi: stl.Expression, w: pd.DataFrame):
    pass
