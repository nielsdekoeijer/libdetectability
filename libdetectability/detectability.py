import numpy as np
import scipy as sp
import torch as tc

from .internal.gammatone_filterbank import gammatone_filterbank
from .internal.outer_middle_ear_filter import outer_middle_ear_filter

class Detectability:
    def __init__(self, frame_size=2048, sampling_rate=48000, taps=32, dbspl=94.0, spl=1.0, relax_threshold=False):
        assert frame_size % 2 == 0, "only evenly-sized frames are supported"
        self.frame_size = frame_size
        self.freq_size = frame_size // 2 + 1
        self.taps = taps
        self.dbspl = dbspl
        self.spl = spl
        self.sampling_rate = sampling_rate

        # prealloc
        self.g = np.power(np.abs(gammatone_filterbank(self.taps, self.frame_size, self.sampling_rate)), 2.0)
        self.h = np.power(np.abs(outer_middle_ear_filter(self.frame_size, self.spl, self.dbspl, self.sampling_rate, relax_threshold=relax_threshold)), 2.0)
        self.leff = min(float(self.frame_size) / float(sampling_rate) / 0.30, 1.0)

        # calibration
        calibration_bin = self.frame_size // 4
        calibration_freq = np.fft.rfftfreq(frame_size, d=1.0/sampling_rate)[calibration_bin]
        A52 = np.power(10.0, (52.0 - (self.dbspl - 20 * np.log10(spl))) / 20.0)
        A70 = np.power(10.0, (70.0 - (self.dbspl - 20 * np.log10(spl))) / 20.0)
        e = A52 * np.sin(2 * np.pi * calibration_freq * np.arange(self.frame_size) / self.sampling_rate)
        x = A70 * np.sin(2 * np.pi * calibration_freq * np.arange(self.frame_size) / self.sampling_rate)
        e = self._spectrum(e)
        x = self._spectrum(x)
        e = self._masker_power_array(e)
        x = self._masker_power_array(x)

        calibration_ca = lambda cs: cs * self.leff * np.sum([self.g[i][calibration_bin] for i in range(self.taps)])
        calibration_bisection = lambda cs: 1.0 - self._detectability(e, x, cs, calibration_ca(cs))
        self.cs = sp.optimize.bisect(calibration_bisection, 1e-2, 1e5)
        self.ca = calibration_ca(self.cs)

    def _spectrum(self, a):
        return np.power(np.abs(np.fft.rfft(a)), 2.0)

    def _masker_power(self, a, i):
        return np.sum(self.h * self.g[i] * a)

    def _masker_power_array(self, a):
        return np.array([self._masker_power(a, i) for i in range(self.taps)])

    def _detectability(self, s, m, cs, ca):
        return cs * self.leff * (s / (m + ca)).sum()

    def frame(self, reference, test):
        assert x.size == self.frame_size and y.size == self.frame_size, f"input frame size different the specified upon construction"
        assert len(x.shape) == 1 and len(y.shape) == 1, f"only support for one-dimensional inputs"

        e = self._spectrum(test - reference)
        x = self._spectrum(reference)
        e = self._masker_power_array(e)
        x = self._masker_power_array(x)

        return self._detectability(e, x, self.cs, self.ca)

class DetectabilityLoss(tc.nn.Module):
    def __init__():
        None
