import numpy as np


class FCLayer(object):
    """
    The layer object of a FCNetwork
    """
    def __init__(self, input_vec_size, output_vec_size, activator):
        """
        initialize a FClayer
        :param input_vec_size:
        :param output_vec_size:
        :param activator:
        """
        self.activator = activator
        self.weight_matrix = np.random.uniform(-1, 1, (output_vec_size, input_vec_size))

    def forward(self, input_vec):
        self.input_vec = input_vec
        self.output_vec = self.activator(np.dot(self.weight_matrix, self.input_vec))
        return self.output_vec

    def backward(self, error_vec):
        self.error_vec = self.input_vec \
                         * (1 - self.input_vec) \
                         * np.dot(self.weight_matrix.T, error_vec)
        self.weight_matrix_grad = -np.dot(self.input_vec.T, error_vec)

    def update(self, rate):
        self.weight_matrix = self.weight_matrix - rate * self.weight_matrix_grad
