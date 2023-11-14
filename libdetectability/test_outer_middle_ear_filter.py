from .outer_middle_ear_filter import outer_middle_ear_filter

def test_outer_middle_ear_filter():
    import matplotlib.pyplot as plt
    filter = outer_middle_ear_filter(2048, 1.0, 94.0, 48000.0, relax_threshold=False)
    plt.plot(filter)
    plt.savefig("outer_middle_ear_filter.png")
