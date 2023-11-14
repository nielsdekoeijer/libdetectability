import numpy as np
import scipy as sp
from scipy.special import factorial, factorial2
from . import *

def gammatone_filterbank(taps, frame_size, sampling_rate):
    # magic number see paper
    sq = np.power(2, 4 - 1)
    f1 = sp.special.factorial(4 - 1)
    f2 = sp.special.factorial2(2 * 4 - 3)
    k = (sq * f1) / (np.pi * f2)

    # centre frequencies + erbs
    f0 = np.array([centre_frequency(tap) for tap in np.arange(taps)])
    erb0 = np.array([freq_as_erb(freq) for freq in f0])
    assert np.max(f0) < sampling_rate / 2, f"Specified taps size yields frequencies above nyquist: {np.max(f0)} > {sampling_rate / 2}"

    return np.array([
            [
                np.power(1.0 + ((f - f0[i]) / (k * erb0[i])), -2.0)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
            for i in range(taps)
        ])
