import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Data
np.random.seed(42)
n_steps = 1000
time = np.linspace(0, 100, n_steps)
# Normal signal + some spikes (incidents)
data = np.sin(time) + np.random.normal(0, 0.1, n_steps)
data[200:210] += 5  # Incident 1
data[600:615] += 5  # Incident 2

# 2. Sliding Window Transformation
def create_sliding_window(series, W, H, threshold=4):
    X, y = [], []
    for i in range(len(series) - W - H):
        # Look-back window
        X.append(series[i : i + W])
        # Look-ahead horizon: check if any value exceeds threshold
        future_window = series[i + W : i + W + H]
        y.append(1 if np.any(future_window > threshold) else 0)
    return np.array(X), np.array(y)

W, H = 20, 10
X, y = create_sliding_window(data, W, H)

# 3. Train/Test Split (No shuffling to preserve time order)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Model Selection: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Prediction & Evaluation
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

print(classification_report(y_test, y_pred))