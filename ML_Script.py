import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from multiprocessing import Process

DATA_LOCATION = 'Data/adult.csv'
SETTING_LOCATION = 'data_settings.txt'

#Function to read settings from data_settings.txt file, starting at line 4
def read_settings():
    settings = {}
    with open(SETTING_LOCATION) as f:
        raw_settings = []
        for line in f.readlines()[3:]:
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

def file_check():
    print('Hello from within thread:', os.getpid())
    latest_settings = read_settings()
    while True:
        current_settings = read_settings()
        print('Settings read, no changes made...')
        if latest_settings != current_settings:
            print('Mismatched detected! Updating settings...')
            latest_settings = current_settings
            adjust_model(latest_settings)

        time.sleep(2)

def adjust_model(params: dict):
    global mlp_clf, X_train, X_test, y_train, y_test
    mlp_clf = MLPClassifier(**params)
    mlp_clf.fit(X_train, y_train)
    mlp_score = mlp_clf.score(X_test, y_test)*100
    print('Accuracy score after adjusting settings:', mlp_score)

######################################################## INITIAL SETUP SECTION ######################################################
##This section establishes a label encoder object, multi-layer perceptron object, and performs the initial training of the ML model##
census_data = pd.read_csv(DATA_LOCATION)    #read data from csv file, store it as a dataframe object
df = pd.DataFrame(census_data)

col_labels = []     #we will store the column labels of our data in col_labels
with open(SETTING_LOCATION) as f:    #open the data_settings text file
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
    watchdog = Process(target=file_check)
    watchdog.start()
    watchdog.join()
    print('done')
    #create thread which determines if change has been made, if so then flip a boolean var / something similar
    #upon flipping variable, main thread updates ML model with new parameters, then flips boolean var again
    #NOTE: keep main thread executing at all times so container doesnt die
    #use multithreading join() function to prevent death of process, if join() doesn't work then look for alternatives
