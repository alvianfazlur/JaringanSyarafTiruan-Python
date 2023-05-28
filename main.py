import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split

# dataset = pd.read_csv('AND.csv')
# dataset = pd.read_csv('OR.csv')
dataset = pd.read_csv('XOR.csv')
data = dataset.iloc[:,0:-1]
label = dataset.iloc[:,-1]

# 1. Single Perceptron
# dataset = pd.read_csv('AND.csv')
# dataset = pd.read_csv('OR.csv')
# dataset = pd.read_csv('XOR.csv')
# data = dataset.iloc[:,0:-1]
# label = dataset.iloc[:,-1]
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(data, label)
# print(clf.score(data,label))
# print(clf.predict(data))

# 3. Multi Layer Perceptron
# model = MLP(hidden_layer_sizes=(2), max_iter=100,activation='relu',learning_rate_init=0.1, solver='sgd')
# model.fit(data,label)
#
# print('score:', model.score(data,label))
# print('predictions:',model.predict(data))
# print('expected:',label)

# 4. Split heart.csv dengan metode Hold-out
dataset = pd.read_csv('heart.csv')
datalabel = dataset.loc[:,['target']]
xtrain, xtest, ytrain, ytest = train_test_split(dataset,datalabel, test_size=0.30, random_state=100)

# 5.
# dataset = xtrain
# data = dataset.iloc[:,0:-1]
# label = dataset.iloc[:,-1]
# model = MLP(hidden_layer_sizes=(4), max_iter=1000,activation='relu',learning_rate_init=0.1, solver='sgd')
# model.fit(data,label)
#
# print('score:', model.score(data,label))
# print('predictions:',model.predict(data))
# print('expected:',label)

# 6.
dataset = xtest
data = dataset.iloc[:,0:-1]
label = dataset.iloc[:,-1]
model = MLP(hidden_layer_sizes=(4), max_iter=1000,activation='relu',learning_rate_init=0.1, solver='sgd')
model.fit(data,label)

print('score:', model.score(data,label))
print('predictions:',model.predict(data))
print('expected:',label)