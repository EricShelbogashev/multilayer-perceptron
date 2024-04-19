import random
import sys
from typing import List, Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

sys.path.append('sample')
from sample.sample_factory import SampleFactory

# %%
# noinspection PyMethodMayBeStatic
class NeuralNetwork:
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], learning_rate: float = 0.002,
                 activation_function: str = 'relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        if activation_function == 'relu':
            self.activation_function = self.relu
            self.activation_function_derive = self.relu_derive
        elif activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_function_derive = self.sigmoid_derive
        elif activation_function == 'tanh':
            self.activation_function = self.tanh
            self.activation_function_derive = self.tanh_derive
        else:
            raise ValueError('Activation function must be either relu or sigmoid or tanh')

        # Инициализация весов и смещений
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            # Случайная инициализация весов с формой (размер_предыдущего_слоя, текущий_слой)
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i + 1]))
            # Инициализация смещений с формой (1, текущий_слой)
            self.biases.append(np.random.randn(1, layer_dims[i + 1]))

    def sigmoid(self, t: np.ndarray) -> np.ndarray:
        # Функция активации Sigmoid: f(x) = 1 / (1 + exp(-x))
        return 1 / (1 + np.exp(-t))

    def sigmoid_derive(self, t: np.ndarray) -> np.ndarray:
        # Производная функции активации Sigmoid: f'(x) = f(x) * (1 - f(x))
        sigmoid_t = self.sigmoid(t)
        return sigmoid_t * (1 - sigmoid_t)

    def tanh(self, t: np.ndarray) -> np.ndarray:
        # Функция активации Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return np.tanh(t)

    def tanh_derive(self, t: np.ndarray) -> np.ndarray:
        # Производная функции активации Tanh: f'(x) = 1 - f(x)^2
        tanh_t = self.tanh(t)
        return 1 - tanh_t ** 2

    def relu(self, t: np.ndarray) -> np.ndarray:
        # Функция активации ReLU: f(x) = max(0, x)
        return np.maximum(0, t)

    def relu_derive(self, t: np.ndarray) -> np.ndarray:
        # Производная функции активации ReLU
        return (t >= 0).astype(float)

    def softmax(self, t: np.ndarray) -> np.ndarray:
        # Функция активации Softmax: exp(x) / sum(exp(x))
        exp = np.exp(t)
        return exp / np.sum(exp)

    def sparse_cross_entropy(self, z_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Функция потерь разреженной перекрестной энтропии
        return -np.sum(y_true * np.log(z_pred))

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in range(len(self.weights) - 1):
            # Прямой проход через скрытые слои с активацией ReLU
            h = self.activation_function(np.dot(h, self.weights[i]) + self.biases[i])
        # Предсказание выходного слоя с активацией Softmax
        t = np.dot(h, self.weights[-1]) + self.biases[-1]
        softmax = self.softmax(t)
        return softmax

    def train(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int) -> List[float]:
        start_time = time.time()
        loss_arr = []
        for epoch in range(epochs):
            random.shuffle(dataset)
            for x, y in dataset:
                # Прямой проход
                h = x
                activations = [h]  # Сохранение активаций для обратного распространения
                for i in range(len(self.weights) - 1):
                    # Прямой проход через скрытые слои с активацией ReLU
                    h = self.activation_function(np.dot(h, self.weights[i]) + self.biases[i])
                    activations.append(h)
                # Прямой проход через выходной слой с активацией Softmax
                t = np.dot(h, self.weights[-1]) + self.biases[-1]
                z = self.softmax(t)
                # Вычисление перекрестной энтропии
                E = self.sparse_cross_entropy(z, y)
                loss_arr.append(E)

                # Обратный проход (обратное распространение)
                dE_dt = z - y  # Вычисление производной потерь по входу выходного слоя
                dE_dw = [activations[-1].T @ dE_dt]  # Вычисление производной потерь по весам
                dE_db = [dE_dt]  # Вычисление производной потерь по смещениям
                for i in range(len(self.weights) - 2, -1, -1):
                    # Обратное распространение ошибок через скрытые слои
                    dE_dt = (dE_dt @ self.weights[i + 1].T) * self.activation_function_derive(
                        np.dot(activations[i], self.weights[i]) + self.biases[i]
                    )
                    # Вычисление производных потерь по весам и смещениям
                    dE_dw.insert(0, activations[i].T @ dE_dt)
                    dE_db.insert(0, dE_dt)
                # Обновление весов и смещений с помощью градиентного спуска
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dE_dw[i]
                    self.biases[i] -= self.learning_rate * np.mean(dE_db[i], axis=0)
        print("--- %s seconds learning time ---" % (time.time() - start_time))
        return loss_arr

    def train_with_batch(self, dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int, batch_size: int = 32) -> List[float]:
        start_time = time.time()
        loss_arr = []
        for epoch in range(epochs):
            num_batches = len(dataset) // batch_size
            for batch_idx in range(num_batches):
                batch_data = dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_x = np.vstack([data[0] for data in batch_data])
                batch_y = np.vstack([data[1] for data in batch_data])

                # Forward
                h = batch_x
                activations = [h]
                for i in range(len(self.weights) - 1):
                    h = self.activation_function(np.dot(h, self.weights[i]) + self.biases[i])
                    activations.append(h)
                t = np.dot(h, self.weights[-1]) + self.biases[-1]
                z = self.softmax(t)

                # Compute
                E = self.sparse_cross_entropy(z, batch_y)
                loss_arr.append(E)

                # Backward
                dE_dt = z - batch_y
                dE_dw = [activations[-1].T @ dE_dt]
                dE_db = [dE_dt]
                for i in range(len(self.weights) - 2, -1, -1):
                    dE_dt = (dE_dt @ self.weights[i + 1].T) * self.activation_function_derive(
                        np.dot(activations[i], self.weights[i]) + self.biases[i]
                    )
                    dE_dw.insert(0, activations[i].T @ dE_dt)
                    dE_db.insert(0, dE_dt)

                # Update
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dE_dw[i] / batch_size
                    self.biases[i] -= self.learning_rate * np.mean(dE_db[i], axis=0) / batch_size
        print("--- %s seconds learning time ---" % (time.time() - start_time))
        return loss_arr


def one_hot_encode(y_position, num_classes):
    return np.eye(num_classes)[y_position]

# %%
## USAGE
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIMS = [3]
LEARNING_RATE = 0.01
EPOCHS_NUM = 100
SAMPLE_TYPE = 'gaussian'
ACTIVATION_FUNCTION = 'relu'
NUM_SAMPLES = 400
ERROR = 0.1

raw_dataset = SampleFactory().generate_samples(SAMPLE_TYPE, NUM_SAMPLES, ERROR)
# Преобразование меток в векторе one-hot
dataset = [(raw_dataset.data[i][None, ...], one_hot_encode(raw_dataset.target[i], OUTPUT_DIM)) for i in
           range(len(raw_dataset.target))]
# Создание и обучение нейронной сети
nn = NeuralNetwork(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, LEARNING_RATE, ACTIVATION_FUNCTION)
losses = nn.train(dataset, EPOCHS_NUM)

# Оценка точности на тестовом наборе данных
accuracy = np.mean([np.argmax(nn.predict(x)) == np.argmax(y) for x, y in dataset])
print('Точность на тестовом наборе данных: {:.2f}%'.format(100 * accuracy))

# Построение графика потерь в процессе обучения
plt.plot(losses)
plt.xlabel('Итерации')
plt.ylabel('Потери')
plt.title('График потерь в процессе обучения')
plt.show()

# %%
### PLOT
import matplotlib.pyplot as plt
import numpy as np


def plot_classification_result(classifier, xlim: tuple = (-0.5, 0.5), ylim: tuple = (-0.5, 0.5),
                               resolution: float = 0.005):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / resolution)),
                         np.linspace(ylim[0], ylim[1], int((ylim[1] - ylim[0]) / resolution)))

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    Z_probs = classifier.predict(grid_points)
    Z_labels = np.argmax(Z_probs, axis=1)
    Z_labels = Z_labels.reshape(xx.shape)

    plt.contourf(xx, yy, Z_labels, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Classification Result')

    plt.scatter(raw_dataset.data[:, 0], raw_dataset.data[:, 1], c=raw_dataset.target, cmap=plt.cm.RdYlBu,
                edgecolors='k')
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


plot_classification_result(nn)

# %%
## CROSS-VALIDATION
# Количество частей для кросс-валидации
NUM_FOLDS = 5

# Создаем объект KFold для разделения данных
kf = KFold(n_splits=NUM_FOLDS)

# Инициализируем список для сохранения точности на каждом наборе данных
accuracies = []

# Перебираем различные разбиения данных
for train_index, test_index in kf.split(dataset):
    # Разбиваем данные на обучающий и тестовый наборы
    train_data = [dataset[i] for i in train_index]
    test_data = [dataset[i] for i in test_index]

    # Создаем и обучаем нейронную сеть
    nn = NeuralNetwork(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS, LEARNING_RATE)
    losses = nn.train_with_batch(train_data, EPOCHS_NUM)

    # Оцениваем точность на тестовом наборе данных
    accuracy = np.mean([np.argmax(nn.predict(x)) == np.argmax(y) for x, y in test_data])
    accuracies.append(accuracy)

# Выводим среднюю точность по всем разбиениям
print('Средняя точность на кросс-валидации: {:.2f}%'.format(100 * np.mean(accuracies)))

# %%
#### РЕЗУЛЬТАТЫ
# %%
# ДЛЯ SPIRAL:
HIDDEN_DIMS = [10,10,10,10,10]
LEARNING_RATE = 0.001
EPOCHS_NUM = 500
SAMPLE_TYPE = 'spiral'
ACTIVATION_FUNCTION = 'relu'
# --- 12.268972158432007 seconds learning time ---
# Точность на тестовом наборе данных: 91.00%
# --- 9.949800968170166 seconds learning time ---
# --- 9.895243883132935 seconds learning time ---
# --- 10.124592065811157 seconds learning time ---
# --- 9.883049011230469 seconds learning time ---
# --- 9.848536968231201 seconds learning time ---
# Средняя точность на кросс-валидации: 93.00%

# %%
# ДЛЯ XOR
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIMS = [5]
LEARNING_RATE = 0.01
EPOCHS_NUM = 100
SAMPLE_TYPE = 'xor'
ACTIVATION_FUNCTION = 'tanh'
NUM_SAMPLES = 400
ERROR = 0.1
# --- 0.9067299365997314 seconds learning time ---
# Точность на тестовом наборе данных: 95.25%
# --- 0.7782669067382812 seconds learning time ---
# --- 0.7392561435699463 seconds learning time ---
# --- 0.7398631572723389 seconds learning time ---
# --- 0.7318058013916016 seconds learning time ---
# --- 0.7478809356689453 seconds learning time ---
# Средняя точность на кросс-валидации: 83.25%
# %%
# ДЛЯ CIRCLE
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIMS = [6]
LEARNING_RATE = 0.01
EPOCHS_NUM = 100
SAMPLE_TYPE = 'circle'
ACTIVATION_FUNCTION = 'tanh'
NUM_SAMPLES = 400
ERROR = 0.1
# Точность на тестовом наборе данных: 94.75%
# --- 0.7706329822540283 seconds learning time ---
# --- 0.7260370254516602 seconds learning time ---
# --- 0.7245819568634033 seconds learning time ---
# --- 0.7242660522460938 seconds learning time ---
# --- 0.7269651889801025 seconds learning time ---
# Средняя точность на кросс-валидации: 90.50%
# %%
# ДЛЯ GAUSSIAN
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIMS = [3]
LEARNING_RATE = 0.01
EPOCHS_NUM = 100
SAMPLE_TYPE = 'gaussian'
ACTIVATION_FUNCTION = 'relu'
NUM_SAMPLES = 400
ERROR = 0.1
# --- 0.9070570468902588 seconds learning time ---
# Точность на тестовом наборе данных: 99.25%
# --- 0.7225029468536377 seconds learning time ---
# --- 0.7151198387145996 seconds learning time ---
# --- 0.7308521270751953 seconds learning time ---
# --- 0.7312939167022705 seconds learning time ---
# --- 0.7284867763519287 seconds learning time ---
# Средняя точность на кросс-валидации: 99.