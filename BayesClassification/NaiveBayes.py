import numpy as np
import math


class NaiveBayes(object):
    """
    朴素贝叶斯分类器实现类
    """

    def __init__(self, label_num):
        """
        初始化函数
        :param label_num:类标号数量
        """
        self.label_num = label_num
        self.records = None

    def train(self, dataset):
        """
        在这里并未训练模型，仅是将不同类标号的记录分开。命名为train是为了和其他模型统一
        :param dataset:
        """
        self.records = self.__separate_records(dataset)

    def __separate_records(self, dataset):
        """
        将不同类的记录分开
        :param dataset:
        :return:
        """
        records = [[] for i in range(self.label_num)]
        labels = []
        for record in dataset:
            if record[-1] not in labels:
                labels.append(record[-1])
            for i in range(labels.__len__()):
                if record[-1] == labels[i]:
                    records[i].append(record)
        return records

    def predict(self, vector):
        """
        预测给定记录的类标号
        :param vector:测试记录的特征向量
        :return:
        """
        data_size = len(self.records) * len(self.records[0])
        predicted_label = None
        possibility = 0

        for each_group in self.records:
            label, possibilities = self.__calculate_possibility(each_group, vector, data_size)
            print(label, possibilities)
            if possibility <= np.array(possibilities).prod():
                possibility = np.array(possibilities).prod()
                predicted_label = label
        return predicted_label

    def __calculate_possibility(self, each_group, vector, data_size):
        """
        计算概率
        :param each_group:每一个类的所有数据
        :param vector:
        :param data_size:
        :return:
        """
        label = each_group[0][-1]
        possibilities = []
        label_possibility = each_group.__len__() / data_size
        possibilities.append(label_possibility)
        for i in range(len(vector)):
            try:
                feature = float(vector[i])
            except:
                feature = vector[i]

            if type(feature) == float:
                possibilities.append(self.__calculate_continuous(each_group, feature, i))
            else:
                possibilities.append(self.__calculate_discrete(each_group, feature, i))
        return label, possibilities

    def __calculate_continuous(self, each_group, feature, i):
        """
        计算连续属性的概率，假设连续属性服从某种概率分布
        :param each_group:
        :param feature:
        :param i:
        :return:
        """
        samples = []
        for record in each_group:
            sample = record[0][i]
            samples.append(sample)
        samples = np.array(samples, np.float_)
        possibility = self.__normal_distribution(samples, feature)
        return possibility

    def __normal_distribution(self, samples, feature):
        """
        正态分布
        :param samples:
        :param feature:
        :return:
        """
        variance = samples.var()
        standard_deviation = math.sqrt(variance)
        mean = samples.mean()

        exponent = -pow(feature - mean, 2) / (2 * variance)
        coefficient = 1 / (math.sqrt(2 * math.pi) * standard_deviation)
        possibility = coefficient * pow(math.e, exponent)
        return possibility

    def __calculate_discrete(self, each_group, feature, i):
        """
        计算离散属性的概率，用频率估计概率
        :param each_group:
        :param feature:
        :param i:
        :return:
        """
        count = 0
        possibility = 0
        for record in each_group:
            vector = record[0]
            if feature == vector[i]:
                count += 1
        if count != 0:
            possibility = count / each_group.__len__()
        return possibility
