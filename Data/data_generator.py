import numpy as np
import pandas as pd

def generate_data(n=10000):
    np.random.seed(42)

    data = np.random.rand(50, 5, n)

    for _ in range(10):
        idx = np.random.randint(100, n - 100)
        data[idx:idx+15] += np.random.uniform(40, 100)

    df = pd.DataFrame({'latency':data})
    df.to_csv('metrics.csv', index = False)
    print("metrics.csv generated")


if __name__ == '__main__':
    generate_data()
