"""
Efficient robust monitoring for STL.

[1] A. Donzé, T. Ferrère, and O. Maler, “Efficient Robust Monitoring for STL,”
    in Computer Aided Verification, 2013, pp. 264–279.
"""

import pandas as pd
import numpy as np

import temporal_logic.signal_tl as stl


BOTTOM = -np.inf
TOP = np.inf


def efficient_robustness(phi: stl.Expression, w: pd.DataFrame):
    pass

    
