'''
Create a large synthetic data for training a neural network
'''

from random import randint, sample
import numpy as np
import matplotlib.pyplot as plt


def generate_int(rand, sample_size, low, high, noise_variance):
    x = rand.randint(low, high, size=(sample_size, 1))
    x += rand.randint(-noise_variance, noise_variance, size=(sample_size, 1))

    return x

def generate_float(rand, sample_size, mean, std, noise_variance):
    x = rand.normal(mean, std, size=(sample_size, 1))
    x += rand.uniform(-noise_variance, noise_variance, size=(sample_size, 1))

    return x

if __name__ == '__main__':
    sample_size = 100000
    noise_col_count = 130
    int_col_ratio = 0.25
    rand = np.random.RandomState(23)

    noise_cols = []

    for _ in range(noise_col_count):
        if rand.random() < int_col_ratio:
            low = rand.randint(0, 40)
            high = low + rand.randint(50, 1000)
            noise_var = rand.randint(1, max(5, (high - low) // 20))
            noise_cols.append(
                generate_int(rand, sample_size, low, high, noise_var)
            )
        else:
            mean = 1.0 * rand.randint(-10, 10)
            std = rand.uniform(0.5, 20)
            noise_var = rand.uniform(0.05, max(0.1, std / 20))
            noise_cols.append(
                generate_float(rand, sample_size, mean, std, noise_var)
            )

    x1 = generate_int(rand, sample_size, 0, 120, 10)
    x2 = generate_int(rand, sample_size, -20, 40, 4)
    x3 = generate_float(rand, sample_size, 0, 5, 0.2)
    x4 = generate_float(rand, sample_size, 10, 5, 1.3)
    x5 = generate_float(rand, sample_size, 0, 20, 2)
    x6 = generate_float(rand, sample_size, 0, 5, 1)
    x7 = generate_float(rand, sample_size, 0, 2, 0.01)
    x8 = generate_float(rand, sample_size, 0, 10, 0.5)
    x9 = generate_int(rand, sample_size, 0, 10, 10)
    x10 = generate_int(rand, sample_size, -100, 250, 4)
    x11 = generate_int(rand, sample_size, 0, 40, 4)

    x12 = generate_int(rand, sample_size, -20, 200, 7)
    x13 = generate_float(rand, sample_size, 5, 50, 4)
    x14 = generate_float(rand, sample_size, 10, 3, 0.01)
    x15 = generate_float(rand, sample_size, 8, 50, 2)
    x16 = generate_float(rand, sample_size, 0, 3, 1)
    x17 = generate_float(rand, sample_size, 0, 1, 0.01)
    x18 = generate_float(rand, sample_size, 0, 10, 1)
    x19 = generate_int(rand, sample_size, 0, 80, 3)
    x20 = generate_int(rand, sample_size, -100, 250, 4)
    data_cols = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                 x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]

    y = (x1.clip(20, 70) // 10) / 5.0
    y *= (x2 < 0).astype('float')
    y += x3 * 0.2
    y = (y - y.mean()) / y.std()
    y = y.clip(-0.5, 1)
    y += x4.clip(2, 30) * 0.2 * (x2 // 10)
    y *= (x5 > -2).astype('float')
    y += x6 * 0.3 + x7 * 0.5 + x8 * 0.2
    y = (y - y.mean()) / y.std()
    y = y.clip(-0.8, 0.3)
    y *= (((x9 > 3).astype(float) + (x10 > 80).astype(float) + (x11 > 10).astype(float)) > 1).astype(float)
    y = (y - y.mean()) / y.std()

    y += x13 * 0.01 + x14 * 0.02 + x15 * 0.05
    y *= x12.clip(20, 150) // 20
    y = (y - y.mean()) / y.std()
    y.clip(-0.8, 0.7)
    y += x16 * 0.3 + x17 * 0.3 + x18 * 0.4
    y *= (y > -1).astype('float')
    y += (x20 > 0).astype('float') * (x19 // 20) * (x20.clip(0, 200) // 20)
    y = (y > 5).astype('float')


    data = np.concatenate(noise_cols + data_cols, axis=-1)

    print(data.shape)
    print(y.sum())

    np.save('X_data.npy', data)
    np.save('y_data.npy', y)
