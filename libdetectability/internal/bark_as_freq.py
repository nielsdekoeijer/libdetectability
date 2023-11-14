import numpy as np

def bark_as_freq(bark):
    return 650.0 * np.sinh(bark / 7.0)
