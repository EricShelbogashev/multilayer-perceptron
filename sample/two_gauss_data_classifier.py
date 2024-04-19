import numpy as np
from sklearn.utils import Bunch


class TwoGaussDataClassifier:
    def generate_samples(self, num_samples=1000, error=0.1):
        variance_scale = np.interp(error, [0, 0.5], [0.5, 4])
        n = num_samples // 2

        positive_samples = np.random.normal(2, variance_scale, size=(n, 2))
        positive_labels = np.ones((n, 1))
        positive_data = np.hstack((positive_samples, positive_labels))

        negative_samples = np.random.normal(-2, variance_scale, size=(n, 2))
        negative_labels = np.zeros((n, 1))
        negative_data = np.hstack((negative_samples, negative_labels))

        combined_data = np.vstack((positive_data, negative_data))
        data_min = np.min(combined_data[:, :2])
        data_max = np.max(combined_data[:, :2])
        combined_data[:, :2] = ((combined_data[:, :2] - data_min) / (data_max - data_min)) - 0.5

        data = combined_data[:, :2]
        target = combined_data[:, 2].astype(int)
        target_names = np.array(['class_0', 'class_1'])
        feature_names = ['feature_1', 'feature_2']
        return Bunch(data=data, target=target, target_names=target_names, feature_names=feature_names)
