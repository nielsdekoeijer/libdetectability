import numpy as np
import tensorflow as tf
import torch
import os

# Import the PyTorch version of the class
from libdetectability.experimental.detectability_loss_l1_det_torch import DetectabilityLossL1Det as TorchDetectabilityLossL1Det
from libdetectability.experimental.detectability_loss_l1_det_tf import DetectabilityLossL1Det as TensorflowDetectabilityLossL1Det


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test_detectability_loss():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)

    # Model parameters
    frame_size = 2048
    sampling_rate = 48000
    taps = 32
    dbspl = 94.0
    spl = 1.0
    threshold_mode = "hearing_regularized"
    normalize_gain = False
    norm = "backward"
    reduction = "mean"
    eps = 1e-8

    # Initialize both models
    torch_model = TorchDetectabilityLossL1Det(
        frame_size=frame_size,
        sampling_rate=sampling_rate,
        taps=taps,
        dbspl=dbspl,
        spl=spl,
        threshold_mode=threshold_mode,
        normalize_gain=normalize_gain,
        norm=norm,
        reduction=reduction,
        eps=eps,
    )

    tf_model = TensorflowDetectabilityLossL1Det(
        frame_size=frame_size,
        sampling_rate=sampling_rate,
        taps=taps,
        dbspl=dbspl,
        spl=spl,
        threshold_mode=threshold_mode,
        normalize_gain=normalize_gain,
        norm=norm,
        reduction=reduction,
        eps=eps,
    )

    # Generate random test data
    batch_size = 4
    reference = np.random.randn(batch_size, frame_size).astype(np.float32)
    test = np.random.randn(batch_size, frame_size).astype(np.float32)

    # Convert data to tensors
    reference_torch = torch.from_numpy(reference)
    test_torch = torch.from_numpy(test)

    reference_tf = tf.convert_to_tensor(reference)
    test_tf = tf.convert_to_tensor(test)

    # Compute outputs
    output_torch = torch_model(reference_torch, test_torch).detach().numpy()
    output_tf = tf_model(reference_tf, test_tf).numpy()

    # Compare outputs
    print("Torch output:", output_torch)
    print("TensorFlow output:", output_tf)

    # Check if outputs are close
    difference = np.abs(output_torch - output_tf)
    print("Difference:", difference)

    assert np.allclose(output_torch, output_tf, atol=1e-5), "Outputs are not the same"

if __name__ == "__main__":
    test_detectability_loss()

