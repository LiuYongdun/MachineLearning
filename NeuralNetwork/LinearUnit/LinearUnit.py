from Perceptron import Perceptron


def activator(x):
    return x


class LinearUnit(Perceptron.Perceptron):
    """
    线性单元类
    """

    def __int__(self, num_dimension):
        Perceptron.Perceptron.__init__(num_dimension, activator)
