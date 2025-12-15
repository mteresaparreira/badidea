import os
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report


df = pd.read_csv('../../data/allParticipants_5fps_downsampled_preprocessed_norm.csv')

results_directory = '../../naiveModels_results/'
    
# Create 'results_directory' if it doesn't exist
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# # Extract features and labels

# # for naive & naive_n datasets
# features = df.iloc[:, 3:]
# target_class = df['class'].values

# # for full & full_n datasets
features = df.iloc[:, 4:]
target_class = df.iloc[:, 2].values
target_class = target_class.astype('int')

# Split the data

X_train, X_test, y_train, y_test = train_test_split(features, target_class, test_size = 0.20, random_state = 42)

# SVM without Grid Search

svm_classifier = svm.SVC(decision_function_shape='ovo')
svm_classifier.fit(X_train, y_train)

y_predict = svm_classifier.predict(X_test)

with open(f'{results_directory}/superBAD_svm.txt', 'a') as output_file:
    output_file.write('Begin Model: SVM without GridSearch' + '\n')
    output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
    output_file.write(classification_report(y_test, y_predict))
    output_file.write("---------END---------")


# SVM with GridSearch

gamma_exp = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
c_exp = [-5,-3,-1,1,3,5,7,9,11,13,15]

gamma_list = []
c_list = []

for i in gamma_exp:
    gamma_list.append(2**i)

for i in c_exp:
    c_list.append(2**i)

# parameters = {'kernel': ['rbf'], 'C': c_list, 'gamma': gamma_list}
parameters = {'kernel': ['rbf', 'sigmoid', 'poly'], 'C': c_list, 'gamma': gamma_list}
grid = GridSearchCV(svm.SVC(decision_function_shape='ovo', random_state = 42), parameters, refit= True, verbose = True)

grid.fit(X_train, y_train)

# print best parameter after tuning
# print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
  
# print classification report
# print(classification_report(y_test, grid_predictions))

with open(f'{results_directory}/superBAD_svm.txt', 'a') as output_file:
    output_file.write('Begin Model: SVM with GridSearch' + '\n')
    output_file.write('---------------------------------------')
    output_file.write('BEST PARAMETERS: ' + '\n')
    output_file.write(grid.best_params_)
    output_file.write('BEST ESTIMATOR: ' + '\n')
    output_file.write(grid.best_estimator_)
    output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
    output_file.write(classification_report(y_test, grid_predictions))