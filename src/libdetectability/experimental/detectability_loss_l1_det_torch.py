import numpy as np
import scipy as sp
import torch as torch

from libdetectability.detectability import Detectability
from libdetectability.internal.gammatone_filterbank import gammatone_filterbank
from libdetectability.internal.outer_middle_ear_filter import outer_middle_ear_filter


class DetectabilityLossL1Det(torch.nn.Module):
    def __init__(
        self,
        frame_size=2048,
        sampling_rate=48000,
        taps=32,
        dbspl=94.0,
        spl=1.0,
        threshold_mode="hearing_regularized",
        normalize_gain=False,
        norm="backward",
        reduction="mean",
        eps=1e-8,
    ):
        super(DetectabilityLossL1Det, self).__init__()
        self.detectability = Detectability(
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            taps=taps,
            dbspl=dbspl,
            spl=spl,
            threshold_mode=threshold_mode,
            normalize_gain=normalize_gain,
            norm=norm,
        )
        self.ca = self.detectability.ca
        self.cs = self.detectability.cs
        self.frame_size = self.detectability.frame_size
        self.taps = self.detectability.taps
        self.leff = self.detectability.leff
        self.norm = self.detectability.norm
        self.h = torch.from_numpy(self.detectability.h)
        self.g = torch.from_numpy(self.detectability.g)
        self.G = torch.from_numpy(self.detectability.h) * torch.from_numpy(
            self.detectability.g
        ).unsqueeze(0)
        self.reduction = reduction
        self.eps = eps
        self.normalize_gain = normalize_gain

    def _spectrum(self, a):
        return torch.pow(torch.abs(torch.fft.rfft(a, axis=1, norm=self.norm)), 2.0)

    def _masker_power_array(self, a):
        return torch.sum(a.unsqueeze(1) * self.G, axis=2)

    def _detectability(self, s, m, cs, ca):
        return cs * self.leff * (s / (m + ca)).sum(axis=1)

    def to(self, device):
        super().to(device)
        self.G = self.G.to(device)
        self.h = self.h.to(device)
        self.g = self.g.to(device)
        return self

    def _frame(self, reference, test):
        assert (
            len(reference.shape) == 2 and len(test.shape) == 2
        ), f"only support for batched one-dimensional inputs"
        assert (
            reference.shape[1] == self.frame_size and test.shape[1] == self.frame_size
        ), f"input frame size different the specified upon construction"

        e = self._spectrum(test - reference)

        # NOTE: this part is critical!
        gain = self._gain(test.detach())
        return torch.norm(gain * e, p=1, dim=1)

    def _gain(self, reference):
        assert (
            len(reference.shape) == 2
        ), f"only support for batched one-dimensional inputs"
        assert (
            reference.shape[1] == self.frame_size
        ), f"input frame size different the specified upon construction"

        x = self._spectrum(reference)
        x = self._masker_power_array(x)
        numer = (self.cs * self.leff * self.h * self.g).unsqueeze(0)
        denom = (x + self.ca).unsqueeze(-1)
        G = numer / denom
        gain = G.sum(axis=1).sqrt()

        if self.normalize_gain:
            factor = torch.norm(gain, p="fro", dim=1).unsqueeze(-1)
            gain = gain / factor

        return gain

    def forward(self, reference, test):
        batches = self._frame(reference, test)

        if self.reduction == "mean":
            return batches.mean()

        if self.reduction == None:
            return batches
