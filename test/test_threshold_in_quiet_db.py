from libdetectability.internal.threshold_in_quiet_db import threshold_in_quiet_db

def test_threshold_in_quiet_db():
    import matplotlib.pyplot as plt
    import numpy as np
    freq = np.fft.rfftfreq(2048, d=1.0/48000.0)
    threshold = np.array([threshold_in_quiet_db(f) for f in freq])
    plt.plot(freq, threshold)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig("test_threshold_in_quiet_db.png")
    plt.close()
