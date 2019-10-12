from SVM import LinearSVM
import numpy as np
from matplotlib import pyplot as plt

dataset = []
with open('./testSet.txt') as file:
    lines = file.readlines()
    for line in lines:
        splits = list(map(lambda x: float(x), line.replace('\n', '').split('\t')))
        vector = splits[:-1]
        label = splits[-1]
        dataset.append([vector, label])

linearSVM = LinearSVM.LinearSVM(dataset.__len__(), dataset[0][0].__len__())
linearSVM.train(dataset, 20)
print(linearSVM)

for record in dataset:
    vector = record[0]
    label = record[-1]
    if label == 1:
        plt.plot(vector[0], vector[1], 'r-o')
    else:
        plt.plot(vector[0], vector[1], 'g-o')

    predict = linearSVM.predict(vector)
    print(record.__str__() + predict.__str__() + '\n')

x1 = np.linspace(0, 10, 1000)
x2 = (-linearSVM.bias - linearSVM.weight_vec[0] * x1) / linearSVM.weight_vec[1]
plt.plot(x1, x2)
plt.show()
