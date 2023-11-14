import numpy as np
from .threshold_in_quiet import threshold_in_quiet

def outer_middle_ear_filter(frame_size, spl, dbspl, sampling_rate, relax_threshold=False):
    if not relax_threshold:
        return np.array([1.0 / threshold_in_quiet(f, spl, dbspl) for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))])
    else:
        threshold = np.array([threshold_in_quiet(f, spl, dbspl) for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))])
        return np.array([1.0 / np.min(threshold) for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))])
