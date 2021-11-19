import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

census_data = pd.read_csv("Data\\adult.csv")
df = pd.DataFrame(census_data)

col_labels = []
model_parameters = []
with open('data_settings.txt') as f:
    lines = f.readlines()
    categories = lines[0].split(':')[1].strip().split(',')
    for category in categories:
        col_labels.append(category)

#col_labels = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','earnings']
enc = LabelEncoder()
for col in col_labels:
    census_data[col] = census_data[col].astype('str')
    census_data[col] = enc.fit_transform(census_data[col])

features = df.iloc[:,:-1]
classification = df.iloc[: , -1]
X_train, X_test, y_train, y_test = train_test_split(features, classification)

#Parameters of the machine learning model go within MLPClassifier()
#Please see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html for details on the multi-layer perceptron model.
mlp_clf = MLPClassifier()
mlp_clf.fit(X_train, y_train)
mlp_score = mlp_clf.score(X_test, y_test)*100
print(mlp_score)

f.close()
