import math

import numpy as np
from sklearn.utils import Bunch


class SpiralSampleGenerator:
    def generate_samples(self, num_samples=1000, error=0.1):
        n = num_samples // 2

        def generate_samples_impl(n: int, noise: float, delta: float, label: int):
            r = np.arange(n) / n * 5
            t = 1.75 * np.arange(n) / n * 2 * np.pi + delta
            x = r * np.sin(t) + noise * np.random.uniform(-math.pi / 2, math.pi / 2, n)
            y = r * np.cos(t) + noise * np.random.uniform(-math.pi / 2, math.pi / 2, n)
            labels = np.full(n, label)
            return np.column_stack((x, y, labels))

        samples1 = generate_samples_impl(n, error, 0, 0)
        samples2 = generate_samples_impl(n, error, np.pi, 1)
        data = np.vstack((samples1, samples2))

        # Normalize the features to the range [-0.5, 0.5]
        data_min = np.min(data[:, :-1], axis=0)
        data_max = np.max(data[:, :-1], axis=0)
        data[:, :-1] = ((data[:, :-1] - data_min) / (data_max - data_min)) - 0.5

        target = data[:, -1].astype(int)
        target_names = np.array(['class_0', 'class_1'])
        feature_names = ['feature_1', 'feature_2']
        return Bunch(data=data[:, :-1], target=target, target_names=target_names, feature_names=feature_names)
