import numpy as np
from .gammatone_filterbank import gammatone_filterbank
from .outer_middle_ear_filter import outer_middle_ear_filter

class Detectability:
    def __init__(self, frame_size=2048, sampling_rate=48000, taps=10, dbspl=94.0, spl=1.0, relax_threshold=False):
        assert frame_size % 2 == 0, "only evenly-sized frames are supported"
        self.frame_size = frame_size
        self.freq_size = frame_size // 2 + 1
        self.taps = taps
        self.dbspl = dbspl
        self.spl = spl
        self.sampling_rate = sampling_rate
        self.g = gammatone_filterbank(self.taps, self.frame_size, self.sampling_rate)
        self.h = outer_middle_ear_filter(self.frame_size, self.spl, self.dbspl, self.sampling_rate, relax_threshold=relax_threshold)
        self.leff = min(float(self.frame_size) / float(sampling_rate) / 0.30, 1.0)
        self.ca = 1.0
        self.cs = 0.0

    def _spectrum(self, a):
        return np.power(np.abs(np.fft.rfft(a)), 2.0)

    def _masker_power(self, a, i):
        return (1.0 / self.frame_size) * np.sum(self.h * self.g[i] * a)

    def _masker_power_array(self, a):
        return np.array([self._masker_power(a, i) for i in range(self.taps)])

    def _detectability(self, s, m):
        return self.cs * self.leff * np.sum(s / (self.frame_size * m + self.ca))

    def frame(self, x, y):
        assert x.size == self.frame_size and y.size == self.frame_size, f"input frame size different the specified upon construction"
        assert len(x.shape) == 1 and len(y.shape) == 1, f"only support for one-dimensional inputs"

        e = self._spectrum(x - y)
        x = self._spectrum(x)
        s = self._masker_power_array(e)
        m = self._masker_power_array(x)
        d = self._detectability(s, m)

    def mean(self, x, y, overlap_size=None):
        if overlap_size == None:
            overlap_size = self.frame_size // 2
        assert (x.size() - self.frame_size) % overlap_size == 0, \
            "overlap_size must be such that input sequences fit exactly within resulting frames"
