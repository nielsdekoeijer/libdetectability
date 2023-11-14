from .erb_as_freq import erb_as_freq
from .freq_as_erb import freq_as_erb

def centre_frequency(tap, base_freq=20.0):
    return erb_as_freq(freq_as_erb(base_freq) + tap)

