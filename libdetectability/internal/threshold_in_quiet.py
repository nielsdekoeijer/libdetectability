import numpy as np
from .threshold_in_quiet_db import threshold_in_quiet_db

def threshold_in_quiet(freq, spl, dbspl):
    offset = dbspl - 20 * np.log10(spl)
    return np.power(10.0, (threshold_in_quiet_db(freq) - offset) / 20.0)
