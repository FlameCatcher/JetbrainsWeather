import numpy as np
import pandas as pd

def generate_data(n=10000):
    np.random.seed(42)

    data = np.random.normal(50, 5, n)

    for _ in range(10):
        idx = np.random.randint(100, n - 100)
        duration = np.random.randint(5, 15)
        magniture = np.random.randint(40, 100)
        data[idx:idx+duration] += magniture

    df = pd.DataFrame({'latency':data.flatten()})
    df.to_csv('metrics.csv', index = False)
    print("metrics.csv generated")


if __name__ == '__main__':
    generate_data()
