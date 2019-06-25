from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from typing import Union, List, Tuple, Optional

import pandas as pd
import numpy as np

import temporal_logic.signal_tl as stl

from temporal_logic.signal_tl.monitors.efficient_robustness import efficient_robustness
from temporal_logic.signal_tl.monitors.lti_semantics import lti_filter_monitor


def eval_bool(phi, w, t=None):
    # type: stl.Expression, pd.DataFrame, Optional[Union[List, Tuple, np.ndarray]] -> pd.Series
    rho = efficient_robustness(phi, w)
    chi = rho > 0
    if t is not None and isinstance(t, (tuple, list, np.ndarray)):
        return chi[t]
    return chi
