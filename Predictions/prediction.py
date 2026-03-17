import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report

W = 30 # Look-back window
H = 10 # Prediction horizon
THRESHOLD = 85 # an "incident"

def prepare_data(file):
    df = pd.read_csv(file)
    values = df['latency'].values
    X, y = [], []

    for i in range(len(values) - W - H):
        X.append(values[i : i + W])
        # incident in next H steps?
        future_spike = np.any(values[i + W : i + W + H] > THRESHOLD)
        y.append(1 if future_spike else 0)

        return np.array(X), np.array(y)

if __name__ == '__main__':
    X, y = prepare_data('metrics.csv')

    # train on first 80%, test on last 20%
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    print("------ Model performance ------ ")
    print(classification_report(y_test, model.predict(X_test)))

