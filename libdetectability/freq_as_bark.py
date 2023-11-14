import numpy as np

def freq_as_bark(freq):
    return 7.0 * np.arcsinh(freq / 650.0)
