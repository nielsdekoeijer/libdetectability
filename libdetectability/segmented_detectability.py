from libdetectability.detectability import Detectability
from libdetectability.detectability_loss import DetectabilityLoss
import libsegmenter
import torch


class SegmentedDetectability:
    def __init__(
        self,
        frame_size=2048,
        sampling_rate=48000,
        taps=32,
        dbspl=94.0,
        spl=1.0,
        relax_threshold=False,
        normalize_gain=False,
        norm="backward",
    ):
        self.detectability = DetectabilityLoss(
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            taps=taps,
            dbspl=dbspl,
            spl=spl,
            relax_threshold=relax_threshold,
            normalize_gain=normalize_gain,
            norm=norm,
        )

        window = libsegmenter.hann(frame_size)
        assert frame_size % 2 == 0, "only evenly-sizes frames are supported"
        hop_size = frame_size // 2
        self.segmenter = libsegmenter.make_segmenter(
            backend="torch",
            frame_size=frame_size,
            hop_size=hop_size,
            window=window,
            mode="wola",
            edge_correction=False,
        )

    def calculate_segment_detectability_relative(self, reference_signal, test_signal):
        assert (
            reference_signal.shape == test_signal.shape
        ), "reference and test signal must be of the same shape"
        reference_segments = self.segmenter.segment(reference_signal)
        test_segments = self.segmenter.segment(test_signal)
        number_of_batch_elements = reference_segments.shape[0]
        number_of_frames = reference_segments.shape[1]
        detectability_of_segments = torch.zeros(
            (number_of_batch_elements, number_of_frames)
        )
        for fIdx in range(0, number_of_frames):
            detectability_of_segments[:, fIdx] = self.detectability.frame(
                reference_segments[:, fIdx], test_segments[:, fIdx]
            )

        return detectability_of_segments

    def calculate_segment_detectability_absolute(self, reference_signal, test_signal):
        assert (
            reference_signal.shape == test_signal.shape
        ), "reference and test signal must be of the same shape"
        reference_segments = self.segmenter.segment(reference_signal)
        test_segments = self.segmenter.segment(test_signal)
        number_of_batch_elements = reference_segments.shape[0]
        number_of_frames = reference_segments.shape[1]
        detectability_of_segments = torch.zeros(
            (number_of_batch_elements, number_of_frames)
        )
        for fIdx in range(0, number_of_frames):
            detectability_of_segments[:, fIdx] = self.detectability.frame_absolute(
                reference_segments[:, fIdx], test_segments[:, fIdx]
            )

        return detectability_of_segments
