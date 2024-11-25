import numpy as np
import tensorflow as tf
import torch

# Import the Detectability class from your modules
from libdetectability.detectability import Detectability
from libdetectability.internal.gammatone_filterbank import gammatone_filterbank
from libdetectability.internal.outer_middle_ear_filter import outer_middle_ear_filter

class DetectabilityLossL1Det(tf.Module):
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
        super().__init__()
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
        self.h = tf.convert_to_tensor(self.detectability.h, dtype=tf.float32)
        self.g = tf.convert_to_tensor(self.detectability.g, dtype=tf.float32)
        self.G = self.h * tf.expand_dims(self.g, axis=0)
        self.reduction = reduction
        self.eps = eps
        self.normalize_gain = normalize_gain

    def _spectrum(self, a):
        N = tf.shape(a)[1]
        fft_a = tf.signal.rfft(a)
        if self.norm == "backward":
            norm_factor = 1.0
        elif self.norm == "forward":
            norm_factor = 1.0 / tf.cast(N, tf.float32)
        elif self.norm == "ortho":
            norm_factor = 1.0 / tf.sqrt(tf.cast(N, tf.float32))
        else:
            raise ValueError(f"Unsupported norm value: {self.norm}")
        fft_a = fft_a * norm_factor
        return tf.math.pow(tf.abs(fft_a), 2.0)

    def _masker_power_array(self, a):
        return tf.reduce_sum(tf.expand_dims(a, axis=1) * self.G, axis=2)

    def _detectability(self, s, m, cs, ca):
        return cs * self.leff * tf.reduce_sum(s / (m + ca), axis=1)

    def _frame(self, reference, test):
        tf.debugging.assert_equal(
            tf.rank(reference), 2, message="Only support for batched one-dimensional inputs"
        )
        tf.debugging.assert_equal(
            tf.shape(reference)[1],
            self.frame_size,
            message="Input frame size differs from the specified upon construction",
        )

        e = self._spectrum(test - reference)
        gain = self._gain(tf.stop_gradient(test))
        return tf.norm(gain * e, ord=1, axis=1)

    def _gain(self, reference):
        tf.debugging.assert_equal(
            tf.rank(reference), 2, message="Only support for batched one-dimensional inputs"
        )
        tf.debugging.assert_equal(
            tf.shape(reference)[1],
            self.frame_size,
            message="Input frame size differs from the specified upon construction",
        )

        x = self._spectrum(reference)
        x = self._masker_power_array(x)
        numer = self.cs * self.leff * self.h * self.g
        numer = tf.expand_dims(numer, axis=0)
        denom = tf.expand_dims(x + self.ca, axis=-1)
        G = numer / denom
        gain = tf.sqrt(tf.reduce_sum(G, axis=1))

        if self.normalize_gain:
            factor = tf.norm(gain, ord='euclidean', axis=1, keepdims=True)
            gain = gain / factor

        return gain

    @tf.function
    def __call__(self, reference, test):
        batches = self._frame(reference, test)

        if self.reduction == "mean":
            return tf.reduce_mean(batches)
        elif self.reduction is None:
            return batches
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

