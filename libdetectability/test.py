from .detectability import Detectability, DetectabilityLoss
import numpy as np
import torch as tc
import pytest

def test_cost():
    npd = Detectability()
    tcd = DetectabilityLoss()

    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)

    npv = npd.frame(x, y)

    x = tc.from_numpy(x).unsqueeze(0)
    y = tc.from_numpy(y).unsqueeze(0)
    x = tc.concatenate((x, x))
    y = tc.concatenate((y, y))

    tcv = tcd.frame(x, y)

    assert tcv[0] == npv
    assert tcv[1] == npv

def test_gain():
    npd = Detectability()

    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)
    npv = npd.frame(x, y)
    g = npd.gain(x)

    assert npv == pytest.approx(np.power(np.linalg.norm(g * np.fft.rfft(x - y)), 2.0))

def test_old():
    import pydetectability as pd
    new = Detectability()
    old = pd.par_model(48000.0, 2048, pd.signal_pressure_mapping(1.0, 94.0))

    x = np.sin(2 * np.pi * 1000 * np.arange(2048) / 48000)
    y = 7.0 * np.sin(2 * np.pi * 1000 * np.arange(2048) / 48000)

    print(old.detectability_gain(x, x - y), 2.0)
    print(new.frame(x, y))
