import numpy as np
import scipy as sp
import torch as tc

from .internal.gammatone_filterbank import gammatone_filterbank
from .internal.outer_middle_ear_filter import outer_middle_ear_filter

class Detectability:
    def __init__(self, frame_size=2048, sampling_rate=48000, taps=64, dbspl=94.0, spl=1.0, relax_threshold=False, norm="backward"):
        assert frame_size % 2 == 0, "only evenly-sized frames are supported"
        self.frame_size = frame_size
        self.freq_size = frame_size // 2 + 1
        self.taps = taps
        self.dbspl = dbspl
        self.spl = spl
        self.sampling_rate = sampling_rate
        self.norm = norm

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
        return np.power(np.abs(np.fft.rfft(a, norm=self.norm)), 2.0)

    def _masker_power(self, a, i):
        return np.sum(self.h * self.g[i] * a)

    def _masker_power_array(self, a):
        return np.array([self._masker_power(a, i) for i in range(self.taps)])

    def _detectability(self, s, m, cs, ca):
        return cs * self.leff * (s / (m + ca)).sum()

    def frame(self, reference, test):
        assert reference.size == self.frame_size and test.size == self.frame_size, f"input frame size different the specified upon construction"
        assert len(reference.shape) == 1 and len(test.shape) == 1, f"only support for one-dimensional inputs"

        e = self._spectrum(test - reference)
        x = self._spectrum(reference)
        e = self._masker_power_array(e)
        x = self._masker_power_array(x)

        return self._detectability(e, x, self.cs, self.ca)

    def gain(self, reference):
        assert reference.size == self.frame_size, f"input frame size different the specified upon construction"
        assert len(reference.shape) == 1, f"only support for one-dimensional inputs"

        x = self._spectrum(reference)
        x = self._masker_power_array(x)
        numer = self.cs * self.leff * self.h * self.g
        denom = (x + self.ca).reshape(-1, 1)
        G = numer / denom
        return np.sqrt(G.sum(axis=0))

class DetectabilityLoss(tc.nn.Module):
    def __init__(self, frame_size=2048, sampling_rate=48000, taps=32, dbspl=94.0, spl=1.0, relax_threshold=True, norm = "backward", reduction="meanlog", eps=1e-8):
        super(DetectabilityLoss, self).__init__()
        self.detectability = Detectability(frame_size=frame_size, sampling_rate=sampling_rate, taps=taps, dbspl=dbspl, \
                spl=spl, relax_threshold=relax_threshold, norm=norm)
        self.ca = self.detectability.ca
        self.cs = self.detectability.cs
        self.frame_size = self.detectability.frame_size
        self.taps = self.detectability.taps
        self.leff = self.detectability.leff
        self.norm = self.detectability.norm
        self.G = tc.from_numpy(self.detectability.h) * tc.from_numpy(self.detectability.g).unsqueeze(0)
        self.reduction = reduction
        self.eps = eps

    def _spectrum(self, a):
        return tc.pow(tc.abs(tc.fft.rfft(a, axis=1, norm=self.norm)), 2.0)

    def _masker_power_array(self, a):
        return tc.sum(a.unsqueeze(1) * self.G, axis=2)

    def _detectability(self, s, m, cs, ca):
        return cs * self.leff * (s / (m + ca)).sum(axis=1)

    def to(self, device):
        super().to(device)
        self.G = self.G.to(device)
        return self

    def frame(self, reference, test):
        assert len(reference.shape) == 2 and len(test.shape) == 2, f"only support for batched one-dimensional inputs"
        assert reference.shape[1] == self.frame_size and test.shape[1] == self.frame_size, f"input frame size different the specified upon construction"

        e = self._spectrum(test - reference)
        x = self._spectrum(reference)
        e = self._masker_power_array(e)
        x = self._masker_power_array(x)

        return self._detectability(e, x, self.cs, self.ca)

    def forward(self, reference, test):
        batches = self.frame(reference, test)
        if self.reduction == "mean":
            return batches.mean()
        if self.reduction == "meanlog":
            batches = tc.log(batches + self.eps)
            return batches.mean()
