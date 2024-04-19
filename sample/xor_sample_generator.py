import numpy as np
from sklearn.utils import Bunch


class XORSampleGenerator:
    def __init__(self, width: float = 1.0, height: float = 1.0):
        self.width = width
        self.height = height

    def generate_samples(self, num_samples=1000, error=0.1):
        x = np.random.uniform(-self.width / 2, self.width / 2, num_samples)
        y = np.random.uniform(-self.height / 2, self.height / 2, num_samples)

        x_error = np.random.uniform(-error, error, num_samples)
        y_error = np.random.uniform(-error, error, num_samples)

        labels = np.logical_xor(x > 0.0, y > 0.0).astype(float)
        x += x_error * (self.width / 2)
        y += y_error * (self.height / 2)

        data = np.column_stack((x, y))
        target = labels.astype(int)
        target_names = np.array(['class_0', 'class_1'])
        feature_names = ['feature_1', 'feature_2']
        return Bunch(data=data, target=target, target_names=target_names, feature_names=feature_names)
