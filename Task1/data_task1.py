import numpy as np
import pandas as pd

def generate_data(n_points=2000):
    time = np.arange(n_points)
    # Normal noise
    signal = np.random.normal(0, 0.5, n_points)
    # Add 5 "incidents" (large spikes)
    incident_indices = [400, 800, 1200, 1600, 1900]
    for idx in incident_indices:
        signal[idx:idx+5] += 10
    return pd.Series(signal)

data = generate_data()