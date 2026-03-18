import numpy as np
import pandas as pd
import joblib
from prediction_system import engineer_features, W, H, Threshold


def calculate_lead_time():
    model = joblib.load('model_artifact.joblib')
    df = pd.read_csv('metrics.csv')
    values = df['latency'].values

    test_start = int(len(values) * 0.8)
    test_data = values[test_start:]

    alerts = []
    actual_incidnets = []

    for i in range(len(test_data) - W - H):
        window = test_data[i:i + W]
        feat = np.array(engineer_features(window).reshape(1, -1))

        prob = model.predict_proba(feat)[0,1]
        if prob > 0.25:
            alerts.append(test_data + i + W)

        if test_data[i + W] > Threshold:
            actual_incidnets.append(test_data + i + W)

        if actual_incidnets and alerts:
            first_incident = actual_incidnets[0]
            first_alert = [a for a in alerts if a < first_incident]
            if first_alert:
                lead = first_incident - first_alert[0]
                print(f" Successful Detection! Lead time : {lead} time steps")
            else:
                print("!!! Incident missed or alert fired to late !!!")


if __name__ == '__main__':
    calculate_lead_time()
