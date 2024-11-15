import numpy as np
from .threshold_in_quiet import threshold_in_quiet


def outer_middle_ear_filter(frame_size, spl, dbspl, sampling_rate, threshold_mode):
    if threshold_mode == "relaxed":
        return np.array(
            [
                1.0 / threshold_in_quiet(f, spl, dbspl)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
        )

    if threshold_mode == "hearing":
        threshold = np.array(
            [
                threshold_in_quiet(f, spl, dbspl)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
        )
        return np.array(
            [
                1.0 / np.min(threshold)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
        )

    if threshold_mode == "hearing_regularized":
        threshold = np.array(
            [
                threshold_in_quiet(f, spl, dbspl, regularized=True)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
        )
        return np.array(
            [
                1.0 / np.min(threshold)
                for f in np.fft.rfftfreq(frame_size, d=(1.0 / sampling_rate))
            ]
        )

    raise (f"Invalid 'threshold_mode' {threshold_mode}")
