import libdetectability as ld
import numpy as np

for N in [512, 1024, 2048, 4096]:
    A52 = np.power(10.0, (52.0 - (94.0 - 20 * np.log10(1.0))) / 20.0) / N
    A70 = np.power(10.0, (70.0 - (94.0 - 20 * np.log10(1.0))) / 20.0) / N
    detectability = ld.Detectability(frame_size=N, relax_threshold=True)
    for f in np.arange(48000 // 2):
        x = A70 / 2 * np.sin(2 * np.pi * f * np.arange(N) / 48000)
        y = x + A52 / 2 * np.sin(2 * np.pi * f * np.arange(N) / 48000)
        print(detectability.frame(x, y))
