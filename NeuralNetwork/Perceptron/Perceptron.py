import numpy as np


class Perceptron(object):
    """
    感知器类
    """
    def __init__(self, num_dimension, activator):
        """
        初始化感知器，将权重向量与偏置项初始化为０
        :param num_dimension: 输入数据的维度
        :param activator: 激活函数，float->float
        """
        self.weight_vector = np.zeros(num_dimension, np.float)
        self.activator = activator
        self.bias = 0.0

    def train(self, record_vectors, num_iteration, rate):
        """
        感知器模型训练入口
        :param record_vectors: 输入向量，以（X，label）形式，其中X是属性向量，label是记录的标签
        :param num_iteration: 迭代次数
        :param rate: 学习速率
        """
        for i in range(num_iteration):
            self.one_iteration(np.array(record_vectors), rate)

    def one_iteration(self, record_vectors, rate):
        """
        一次迭代过程，将所有输入向量用于训练模型
        :param record_vectors: 输入向量
        :param rate: 学习速率
        """
        for record_vector in record_vectors:
            real_label = record_vector[-1]
            x_vector = record_vector[:-1]
            predict_label = self.predict(x_vector)
            self.update(real_label, predict_label, x_vector, rate)

    def predict(self, x_vector):
        """
        模型预测值计算函数
        :param x_vector: 属性向量
        :return: predict_label预测值
        """
        weighted_sum = np.dot(self.weight_vector, x_vector) + self.bias
        predict_label = self.activator(weighted_sum)
        return predict_label

    def update(self, real_label, predict_label, x_vector, rate):
        """
        更新函数，更新权重向量与偏置项
        :param real_label: 实际标签值
        :param predict_label: 预测标签值
        :param x_vector: 属性向量
        :param rate: 学习速率
        """
        delta = real_label - predict_label
        self.weight_vector = self.weight_vector + rate * delta * x_vector
        self.bias = self.bias + rate * delta
