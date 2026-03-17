import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score


W = 30 # Window
H = 10 # Horizon
Threshold = 85 # Incident

def engineer_features(window):
    """
    Translates raw metrics into system insights
    """

    return [
        np.mean(window),            # Average load
        np.std(window),             # Turbulence
        window[-1],                 # Current value
        window[-1] - window[0],     # Trend
        np.max(window)              # Peak in window
    ]

def prepare_data(file):
    df = pd.read_csv(file)
    values = df['latency'].values
    X, y = [], []

    for i in range(len(values) - W - H):
        window =values[i : i + W]
        X.append(engineer_features(window))

        future = values[i + W : i + W + H]
        y.append(1 if np.any(future > Threshold) else 0)

    return np.array(X), np.array(y)


def train_system():

    return
