from libdetectability.internal.gammatone_filterbank import gammatone_filterbank

def test_gammatone_filterbank():
    import matplotlib.pyplot as plt
    import numpy as np
    N = 2048 * 8
    t = 58
    freq = np.fft.rfftfreq(N, d=1.0/48000.0)
    gammatone = gammatone_filterbank(t, N, 48000)
    [plt.plot(freq, gammatone[i]) for i in range(t)]
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig("test_gammatone_filterbank.png")
    plt.close()
