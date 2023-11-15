import numpy as np

def erbs_as_freq(erbs: float):
    return (np.power(10, (erbs / 21.4)) - 1) / 0.00437
