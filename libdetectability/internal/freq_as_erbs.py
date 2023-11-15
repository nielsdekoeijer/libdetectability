import numpy as np

def freq_as_erbs(freq: float):
    return 21.4 * np.log10(1 + 0.00437 * freq)
