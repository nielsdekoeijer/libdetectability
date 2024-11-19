import numpy as np


def threshold_in_quiet_db(freq, regularized=False):
    with np.errstate(divide='ignore'):
        value = (
            3.64 * np.power(freq / 1000.0, -0.8)
            - 6.5 * np.exp(-0.6 * np.power(freq / 1000.0 - 3.3, 2))
            + 10e-4 * np.power(freq / 1000.0, 4)
        )

    if regularized:
        return np.maximum(value, 30)
    else:
        return value
