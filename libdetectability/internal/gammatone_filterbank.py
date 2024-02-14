import numpy as np
import scipy as sp
from scipy.special import factorial, factorial2
from .centre_frequency import centre_frequency
from .freq_as_erbs import freq_as_erbs
from .freq_as_erb import freq_as_erb
from .erbs_as_freq import erbs_as_freq

def gammatone_filterbank(taps, frame_size, sampling_rate):
    # magic number see paper
    sq = np.power(2, 4 - 1)
    f1 = sp.special.factorial(4 - 1)
    f2 = sp.special.factorial2(2 * 4 - 3)
    k = (sq * f1) / (np.pi * f2)

    # Centre frequencies, from the original paper:
    # 
    #   "The centre frequencies of the filters are uniformly spaced on an ERB-rate scale 
    #       and follow the bandwidths as specified by the ERB scale [31]."
    #
    # This leaves room for some ambiguity in my view. E.g. 
    # - uniformly spaced between what frequencies?
    # - how many filters should you use?
    #
    # Previously I have worked with the following definition, spacing them as specified betwen 50 and 0.9 * fs
    # This is a similar approach as taken in in the auditory modeling toolbox (erbspacebw).
    # This is where I specifically take some liberties and welcome corrections...
    f0 = np.array([
        centre_frequency(tap) for tap in np.linspace(freq_as_erbs(50), freq_as_erbs((sampling_rate // 2) * 0.9), taps)
    ])

    # ERB = Equivalent Rectangular Band, essentially, the bandwidth of the filter depends on the frequency
    erb0 = np.array([freq_as_erb(freq) for freq in f0])
    assert np.max(f0) < sampling_rate / 2, \
            f"Specified taps size yields frequencies above nyquist: {np.max(f0)} > {sampling_rate / 2}"
    return np.array([
            [
                np.power(1.0 + np.power((f - f0[i]) / (k * erb0[i]), 2.0), -2.0)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
            for i in range(taps)
        ])
