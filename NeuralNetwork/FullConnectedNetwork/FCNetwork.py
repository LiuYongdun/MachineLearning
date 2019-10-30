from FullConnectedNetwork import FCLayer
import math
import numpy as np


def sigmoid(x):
    return 1.0 / (pow(math.e, -x) + 1)


class FCNetwork(object):

    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = FCLayer.FCLayer(layer_sizes[i], layer_sizes[i + 1], sigmoid)
            self.layers.append(layer)

    def train(self, x_vectors, label_vectors, num_iteration, rate):
        x_vectors = self.add_bias(np.array(x_vectors))
        label_vectors = np.array(label_vectors)
        for i in range(num_iteration):
            self.one_iteration(x_vectors, label_vectors, rate)

    def add_bias(self, x_vectors):
        return x_vectors + 1

    def one_iteration(self, x_vectors, label_vectors, rate):
        for x_vector, label_vector in zip(x_vectors, label_vectors):
            self.predict(x_vector)
            self.calculate_error_vec(label_vector)
            self.update(rate)

    def predict(self, x_vector):
        output_vec = np.array(x_vector)
        for layer in self.layers:
            output_vec = layer.forward(output_vec)
        return output_vec

    def calculate_error_vec(self, label_vector):
        error_vec = (label_vector - self.layers[-1].output_vec) \
                    * self.layers[-1].output_vec \
                    * (1 - self.layers[-1].output_vec)
        for layer in self.layers[::-1]:
            layer.backward(error_vec)
            error_vec = layer.error_vec

    def update(self, rate):
        for layer in self.layers:
            layer.update(rate)
