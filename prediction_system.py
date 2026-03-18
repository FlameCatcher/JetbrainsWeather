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
    X, y = prepare_data('metrics.csv')
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    print("Training the model...")
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_train)[:,1]

    best_t = 0.5
    for t in np.arange(0.1,0.6,0.05):
        y_pred_t = (y_probs > t).astype(int)
        if  recall_score(y_train, y_pred_t) >= 0.8:
            best_t = t
            break

    final_preds = (y_probs > best_t).astype(int)
    print(f"----- Systm Performance (Selected Threshold: {best_t:.2f}) -----)")
    print(classification_report(y_train, final_preds))

    joblib.dump(model, 'model_artifact.joblib')
    return model, best_t


if __name__ == '__main__':
    train_system()