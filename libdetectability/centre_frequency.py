from .bark_as_freq import bark_as_freq
from .freq_as_bark import freq_as_bark

def centre_frequency(tap, base_freq=80.0, bark_skip=0.5):
    return bark_as_freq(freq_as_bark(base_freq) + tap * bark_skip)

