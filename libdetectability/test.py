from .detectability import Detectability, DetectabilityLoss
import numpy as np
import torch as tc
import pytest

def test_cost():
    npd = Detectability(norm="ortho")
    tcd = DetectabilityLoss(norm="ortho")

    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)

    npv = npd.frame(x, y)

    x = tc.from_numpy(x).unsqueeze(0)
    y = tc.from_numpy(y).unsqueeze(0)
    x = tc.concatenate((x, x))
    y = tc.concatenate((y, y))

    tcv = tcd.frame(x, y)

    assert tcv[0] == pytest.approx(npv)
    assert tcv[1] == pytest.approx(npv)

def test_gain():
    npd = Detectability(norm="ortho")

    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)
    npv = npd.frame(x, y)
    g = npd.gain(x)

    assert npv == pytest.approx(np.power(np.linalg.norm(g * np.fft.rfft(x - y, norm="ortho")), 2.0))

def test_cost_old():
    import pydetectability as pd
    new = Detectability()
    old = pd.par_model(48000.0, 2048, pd.signal_pressure_mapping(1.0, 94.0))

    x = np.sin(2 * np.pi * 1000 * np.arange(2048) / 48000)
    y = 7.0 * np.sin(2 * np.pi * 1000 * np.arange(2048) / 48000)

    print(old.detectability_gain(x, x - y), 2.0)
    print(new.frame(x, y))
    print("end")

def test_gain_old():
    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)

    new = Detectability()
    g = new.gain(x)
    print(pytest.approx(np.power(np.linalg.norm(g * np.fft.rfft(x - y)), 2.0)))

    import pydetectability as pd
    old = pd.par_model(48000, 2048, pd.signal_pressure_mapping(1.0, 94.0))
    g = old.gain(x)
    print(pytest.approx(np.power(np.linalg.norm(g * np.fft.rfft(x - y)), 2.0)))
    print("end")
    
