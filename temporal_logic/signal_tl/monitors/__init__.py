from .efficient_robustness import efficient_robustness
import temporal_logic.signal_tl as stl
import pandas as pd

def eval_bool(phi: stl.Expression, w: pd.DataFrame, t=None) -> pd.Series:
    rho = efficient_robustness(phi, w)
    chi = rho > 0
    if t is not None and isinstance(t, (tuple, list)):
        return chi[t]
    return chi
