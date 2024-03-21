import numpy as np
import pandas as pd

def get_scale_weight(y:np.array)->float:
    vc = y.value_counts()
    return vc.max()/vc.min()