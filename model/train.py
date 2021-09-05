import pandas as pd
from sklearn.base import _pprint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

dataUrl = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(dataUrl, names = names)
array = df.values

X = array[: , 0:8]
Y = array[: , 8 ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 101)

model = LogisticRegression()

model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print(result)


joblib.dump(model, 'diab_79.pkl')