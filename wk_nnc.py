import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random

data = pd.read_csv("1.txt", sep = ",", header = None)
del data[0]
del data[1]

test = pd.read_excel("2.xlsx", header = None)
del test[0]
del test [1]


train_price = data[20]
train_price=train_price
del data[20]
test_price = test[20]
test_price=test_price.values
del test[20]
train_scaled=(data-data.min())/(data.max()-data.min())
train_scaled=train_scaled.values
test_scaled=(test-test.min())/(test.max()-test.min())
test_scaled=test.values

tree = KDTree(train_scaled)
nearest_dist, nearest_ind = tree.query(test_scaled[13].reshape(1, -1), k = 3)


def haha1(niubi, num = 1.0, niubi1 = 0.1):
    return num / (niubi + niubi1)
  

def wk_nnc(kdtree, test_point, target, k = 25,
                weight_fun = haha1):
    nearest_dist, nearest_ind = kdtree.query(test_point, k = k)
    avg = 0.0
    totalniubi = 0.0
    for i in range(k):
        niubi = nearest_dist[0][i]
        idx = nearest_ind[0][i]
        niubi1 = weight_fun(niubi)
        avg += niubi1 * target[idx]
        totalniubi += niubi1
    avg = round(avg / totalniubi)
    return avg

def testalgorithm(algo, kdtree, testset, target, test_target):
    error = 0.0
    for row in range(len(testset)):
        guess = algo(kdtree, testset[row].reshape(1, -1), target)
        error += (test_target[row] - guess) ** 2
    return round(np.sqrt(error / len(testset)))
    
random.seed(1191)
ex = random.sample(range(len(test_price)), 5)
print("predicted",";", "actual", " ;", "error")
for i in ex:
    res = wk_nnc(tree, test_scaled[i].reshape(1, -1), train_price)
    print(res,
         " ;", 
         test_price[i],
         " ;",
         abs(test_price[i] - res))

print(testalgorithm(wk_nnc, tree, test_scaled, train_price, test_price)) 

