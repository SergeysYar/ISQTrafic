import numpy as np


def moving_average_forecast(history, window=5):
    """Simple baseline forecast: moving average of last `window` timesteps.

    history: iterable of recent flow counts (ints or floats)
    Returns: predicted next-step flow (float)
    """
    history = list(history)
    if len(history) == 0:
        return 0.0
    window = max(1, min(window, len(history)))
    return float(np.mean(history[-window:]))


if __name__ == "__main__":
    h = [3, 2, 4, 5, 6, 7]
    print("Forecast:", moving_average_forecast(h, window=3))
