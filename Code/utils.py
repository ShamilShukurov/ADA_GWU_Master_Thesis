import numpy as np
import pandas as pd

def get_scale_weight(y:np.array)->float:
    """Utility function used for determining xgboost scale_pos_weight"""
    vc = y.value_counts()
    return vc.max()/vc.min()