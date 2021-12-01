import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#Function to read settings from data_settings.txt file, starting at line 4
def read_settings():
    settings = {}
    with open('data_settings.txt') as f:
        raw_settings = []
        for line in f.readlines()[4:]:
            entry = line.strip().split(':')
            raw_settings.append(entry[0])
            if '.' in entry[1]:
                entry[1] = float(entry[1])
            else:
                try:
                    entry[1] = int(entry[1])
                except ValueError:
                    entry[1] = str(entry[1])
            raw_settings.append(entry[1])
        settings = {raw_settings[i]:raw_settings[i+1] for i in range(0, len(raw_settings), 2)}
    return settings

######################################################## INITIAL SETUP SECTION ######################################################
##This section establishes a label encoder object, multi-layer perceptron object, and performs the initial training of the ML model##
census_data = pd.read_csv("Data\\adult.csv")    #read data from csv file, store it as a dataframe object
df = pd.DataFrame(census_data)

col_labels = []     #we will store the column labels of our data in col_labels
with open('data_settings.txt') as f:    #open the data_settings text file
    lines = f.readlines()       #read all lines and store it in a list object called 'lines'
    col_labels = lines[0].split(':')[1].strip().split(',')  #read first line of data_settings and clean it to have our column labels, then store it in col_labels

enc = LabelEncoder()    #create LabelEncoder object, this will switch all non-numeric categories into numbers for our machine learning model to understand
for col in col_labels:  #go through each column of our data
    census_data[col] = census_data[col].astype('str')   #switch entire column of data in dataframe into type string
    census_data[col] = enc.fit_transform(census_data[col])  #fit the encoder and transform the data, resulting in numeric categories (ex: 'red' -> 1, 'blue' -> 2)

features = df.iloc[:,:-1]   #obtain only the features of the data, which will be all data except the last column
classification = df.iloc[:,-1]  #obtain only the classification of the data, which which will be the last column
X_train, X_test, y_train, y_test = train_test_split(features, classification)   #split our data into two sets of data: training and testing

#Parameters of the machine learning model go within MLPClassifier()
#Please see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html for details on the multi-layer perceptron model
params = read_settings()    #Read the settings from our data_settings document. document contains adjustable settings for our multi-layer perceptron model
mlp_clf = MLPClassifier(**params)   #submit the settings as a dictionary of parameters
mlp_clf.fit(X_train, y_train)   #train the machine learning model with our training sets
############################################################ END OF SETUP ##########################################################


if __name__ == '__main__':
    mlp_score = mlp_clf.score(X_test, y_test)*100
    print('Multi-Layer Perceptron accuracy score with default settings:', mlp_score)
    read_settings()
    with open('data_settings.txt') as f:
        pass
