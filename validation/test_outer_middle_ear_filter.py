from libdetectability.internal.outer_middle_ear_filter import outer_middle_ear_filter


def test_outer_middle_ear_filter():
    import matplotlib.pyplot as plt

    filter = outer_middle_ear_filter(2048, 1.0, 94.0, 48000.0, threshold_mode="hearing")
    plt.plot(filter)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("test_outer_middle_ear_filter.png")
    plt.close()


def test_outer_middle_ear_filter_relaxed():
    import matplotlib.pyplot as plt

    filter = outer_middle_ear_filter(2048, 1.0, 94.0, 48000.0, threshold_mode="relaxed")
    plt.plot(filter)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("test_outer_middle_ear_filter_relaxed.png")
    plt.close()


def test_outer_middle_ear_filter_hearing_regularized():
    import matplotlib.pyplot as plt

    filter = outer_middle_ear_filter(
        2048, 1.0, 94.0, 48000.0, threshold_mode="hearing_regularized"
    )
    plt.plot(filter)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("test_outer_middle_ear_filter_hearing_regularized.png")
    plt.close()
