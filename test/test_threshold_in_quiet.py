from libdetectability.internal.threshold_in_quiet import threshold_in_quiet

def test_threshold_in_quiet():
    import matplotlib.pyplot as plt
    import numpy as np
    freq = np.fft.rfftfreq(2048, d=1.0/48000.0)
    threshold = np.array([threshold_in_quiet(f, 1.0, 94.0) for f in freq])
    plt.loglog(freq, threshold)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig("test_threshold_in_quiet.png")
    plt.close()
