import numpy as np
import math


# def kernel_function(vector_i, vector_j):
#     """
#     多项式核函数，K（X1，X2）=(X1*X2+1)^p
#     """
#     return pow(np.dot(vector_i,vector_j)+1,2)

def kernel_function(vector_i, vector_j):
    """
    高斯核函数（RBF核函数），K（X1，X2）=e^(-(X1-X2)^2/(2*σ^2))
    """
    exponent = -np.dot(vector_i - vector_j, vector_i - vector_j) / (2 * pow(0.5, 2))
    return pow(math.e, exponent)


class NonLinearSVM(object):
    """
    非线性SVM的实现类
    """

    def __init__(self, dataset_size):
        """
        初始化函数
        :param dataset_size:数据集规模，用于初始化 ‘拉格朗日乘子’ 的数量
        """
        self.__multipliers = np.zeros(dataset_size, np.float_)
        self.bias = 0

    def train(self, dataset, iteration_num):
        """
        训练函数
        :param dataset:数据集，每条数据的形式为（X，y），其中X是特征向量，y是类标号
        :param iteration_num:
        """
        dataset = np.array(dataset)
        for k in range(iteration_num):
            self.__update(dataset, k)

    def __update(self, dataset, k):
        """
        更新函数
        :param dataset:
        :param k:
        """
        for i in range(dataset.__len__() // 2):
            j = (dataset.__len__() // 2 + i + k) % dataset.__len__()
            record_i = dataset[i]
            record_j = dataset[j]
            self.__sequential_minimal_optimization(dataset, record_i, record_j, i, j)
            self.__update_bias(dataset)

    def __sequential_minimal_optimization(self, dataset, record_i, record_j, i, j):
        """
        SMO函数，每次选取两条记录，更新对应的‘拉格朗日乘子’
        :param dataset:
        :param record_i:记录i
        :param record_j:记录j
        :param i:
        :param j:
        """
        label_i = record_i[-1]
        vector_i = np.array(record_i[0])
        label_j = record_j[-1]
        vector_j = np.array(record_j[0])

        # 计算出截断前的记录i的‘拉格朗日乘子’unclipped_i
        error_i = self.__calculate_W_transformedX(dataset, vector_i) + self.bias - label_i
        error_j = self.__calculate_W_transformedX(dataset, vector_j) + self.bias - label_j
        eta = kernel_function(vector_i, vector_i) - 2 * kernel_function(vector_i, vector_j) + kernel_function(vector_j,
                                                                                                              vector_j)
        unclipped_i = self.__multipliers[i] + label_i * (error_j - error_i) / eta

        # 截断记录i的`拉格朗日乘子`并计算记录j的`拉格朗日乘子`
        constant = -self.__calculate_constant(dataset, i, j)
        multiplier = self.__quadratic_programming(unclipped_i, label_i, label_j, i, j)
        if multiplier >= 0:
            self.__multipliers[i] = multiplier
            self.__multipliers[j] = (constant - multiplier * label_i) * label_j

    def __update_bias(self, dataset):
        """
        计算偏置项bias，使用平均值作为最终结果
        :param dataset:
        """
        sum_bias = 0
        count = 0
        for k in range(self.__multipliers.__len__()):
            if self.__multipliers[k] != 0:
                label = dataset[k][-1]
                vector = np.array(dataset[k][0])
                sum_bias += label - self.__calculate_W_transformedX(dataset, vector)
                count += 1
        if count == 0:
            self.bias = 0
        else:
            self.bias = sum_bias / count

    def __calculate_W_transformedX(self, dataset, vector_x):
        """
        计算 W * φ(X)
        :param dataset:
        """
        sum = 0
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            vector = np.array(dataset[k][0])
            sum += self.__multipliers[k] * label * kernel_function(vector, vector_x)
        return sum

    def __calculate_constant(self, dataset, i, j):
        label_i = dataset[i][-1]
        label_j = dataset[j][-1]
        dataset[i][-1] = 0
        dataset[j][-1] = 0
        sum_constant = 0
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            sum_constant += self.__multipliers[k] * label
        dataset[i][-1] = label_i
        dataset[j][-1] = label_j
        return sum_constant

    def __quadratic_programming(self, unclipped_i, label_i, label_j, i, j, c=100):
        """
        二次规划，截断`拉格朗日乘子`
        :param unclipped_i:
        :param label_i:
        :param label_j:
        :return:
        """

        multiplier = -1

        """
        硬边缘（hard margin）的二次规划，此时SVM的训练误差为0，模型对数据过拟合程度高，容易受离群点的影响
        """
        # if label_i * label_j == 1:
        #     boundary = self.__multipliers[i] + self.__multipliers[j]
        #     if boundary >= 0:
        #         if unclipped_i <= 0:
        #             multiplier = 0
        #         elif unclipped_i < boundary:
        #             multiplier = unclipped_i
        #         else:
        #             multiplier = boundary
        # else:
        #     boundary = max(0, self.__multipliers[i] - self.__multipliers[j])
        #     if unclipped_i <= boundary:
        #         multiplier = boundary
        #     else:
        #         multiplier = unclipped_i

        """
        软边缘（soft margin）的二次规划，此时SVM有一定的训练误差，模型对数据过拟合程度低，可避免离群点的影响
        """
        if label_i * label_j == 1:
            low_boundary = max(0, self.__multipliers[i] + self.__multipliers[j] - c)
            high_boundary = min(c, self.__multipliers[i] + self.__multipliers[j])
            if unclipped_i <= low_boundary:
                multiplier = low_boundary
            elif unclipped_i < high_boundary:
                multiplier = unclipped_i
            else:
                multiplier = high_boundary
        else:
            low_boundary = max(0, self.__multipliers[i] - self.__multipliers[j])
            high_boundary = min(c, c + self.__multipliers[i] - self.__multipliers[j])
            if unclipped_i <= low_boundary:
                multiplier = low_boundary
            elif unclipped_i < high_boundary:
                multiplier = unclipped_i
            else:
                multiplier = high_boundary

        return multiplier

    def predict(self, dataset, vector):
        result = self.__calculate_W_transformedX(dataset, np.array(vector)) + self.bias
        if result >= 0:
            return 1
        else:
            return -1

    def __str__(self):
        return "multipliers:" + self.__multipliers.__str__() + '\n' + \
               "bias:" + self.bias.__str__()
