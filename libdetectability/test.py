from .detectability import Detectability, DetectabilityLoss
import numpy as np
import torch as tc

def test_cost():
    npd = Detectability()
    tcd = DetectabilityLoss()

    x = np.sin(2 * np.pi * 5.0 * np.arange(2048) / 2048)
    y = np.sin(2 * np.pi * np.arange(2048) / 2048)

    npv = npd.frame(x, y)
    tcv = tcd.forward(tc.from_numpy(x).unsqueeze(0), tc.from_numpy(y).unsqueeze(0))

    assert tcv == npv

