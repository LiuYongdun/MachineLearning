from BayesClassification import NaiveBayes

dataset = []
with open('./testSet.txt') as file:
    lines = file.readlines()
    for line in lines:
        splits = list(line.replace('\n', '').split('\t'))
        vector = splits[:-1]
        label = splits[-1]
        dataset.append([vector, label])

naiveBayes=NaiveBayes.NaiveBayes(2)
naiveBayes.train(dataset)
print(naiveBayes.predict([3,5]))