from SVM import NonLinearSVM
from matplotlib import pyplot as plt

dataset = []
with open('./testSetRBF.txt') as file:
    lines = file.readlines()
    for line in lines:
        splits = list(map(lambda x: float(x), line.replace('\n', '').split('\t')))
        vector = splits[:-1]
        label = splits[-1]
        dataset.append([vector, label])

nonLinearSVM = NonLinearSVM.NonLinearSVM(dataset.__len__())
nonLinearSVM.train(dataset, 100)

count=0
for record in dataset:
    vector = record[0]
    label = record[-1]
    if label == 1:
        plt.plot(vector[0], vector[1], 'r-o')
    else:
        plt.plot(vector[0], vector[1], 'g-o')
    predict = nonLinearSVM.predict(dataset,vector)
    if predict==label:
        count+=1

    print(record.__str__() + predict.__str__() + '\n')

precision=count/dataset.__len__()

print("precison",precision)

plt.show()
