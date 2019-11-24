import pandas as pd
import numpy as np
import random

target_options = [0, 1]
target_array = [random.choice(target_options) for i in range(0,30)]
target = pd.DataFrame(data=target_array, columns=['target'])
#print(target)

data_options = [0, 1, 2, 3]
data_array = []
for i in range(0,30):
    row = []
    row.append([[random.choice(data_options) for i in range(0,5)] for i in range(0,5)]) # code
    row.append([[random.choice(data_options) for i in range(0,5)] for i in range(0,5)]) # numeric
    row.append([random.choice(data_options) for i in range(0,5)]) # to_event
    data_array.append(row)
data = pd.DataFrame(data=data_array, columns=['codes', 'numerics', 'to_event'])
#print(data)

dictionary_array = [
    [0, "Anxiety"],
    [1, "Depression"],
    [2, "ADHD"],
    [3, "Hunger"],
]
dictionary = pd.DataFrame(data=dictionary_array)
#print(dictionary)

pd.to_pickle(data, "data_train.pkl")
pd.to_pickle(data, "data_test.pkl")
pd.to_pickle(target, "target_train.pkl")
pd.to_pickle(target, "target_test.pkl")
pd.to_pickle(dictionary, "dictionary.pkl")