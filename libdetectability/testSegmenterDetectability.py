from .segmentedDetectability import segmented_detectability
import torch as tc
import libsegmenter
import pytest 
import numpy as np

def test_segmenter_detectability_relative():
    sampling_rate = 48000
    window_size = 2048
    hop_size = window_size // 2;

    signal_length = 4*window_size+10
    batch_size = 3
    input = tc.ones((batch_size, signal_length))
    input2 = tc.ones((batch_size, signal_length))

    segmenter = segmented_detectability(frame_size = window_size, sampling_rate = sampling_rate)

    results = segmenter.calculate_segment_detectability_relative(input, input2)
    max_detectability = results.numpy()
    test_value = np.max(max_detectability)
    ref_value = 0.0


    assert test_value == pytest.approx(ref_value)

def test_segmenter_detectability_absolute():
    sampling_rate = 48000
    window_size = 2048
    hop_size = window_size // 2;

    signal_length = 4*window_size+10
    batch_size = 3
    input = tc.ones((batch_size, signal_length))
    input2 = tc.zeros((batch_size, signal_length))

    segmenter = segmented_detectability(frame_size = window_size, sampling_rate = sampling_rate)

    results = segmenter.calculate_segment_detectability_absolute(input, input2)
    max_detectability = results.numpy()
    test_value = np.max(max_detectability)
    ref_value = 0.0


    assert test_value == pytest.approx(ref_value)

    
