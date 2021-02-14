import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


# X
training_data = pd.read_csv(os.path.join(
    os.getcwd(), "data/Features_training_15122020_19122020_Merged_adithya.csv"))
# TODO change x_test
val_data = pd.read_csv(os.path.join(
    os.getcwd(), "data/Features_validation_15122020.csv"))

test_data = pd.read_csv(os.path.join(
    os.getcwd(), "data/features_test_19122020_Final.csv"))

# Y
training_labels = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/label_groundtruth_train_s.csv"))
# Y_test
val_labels = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/label_groundtruth_validation.csv"))

X_labels = ['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
            't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
            'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
            'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

Y_labels = ['label']


# Standardlize data

scaler = StandardScaler()
X_Train = scaler.fit(training_data[X_labels])
X_Train = scaler.transform(training_data[X_labels])

X_Train = pd.DataFrame(X_Train, columns=['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
                                         't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
                                         'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
                                         'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'])

X_Train = pd.concat([training_data['imageID'], X_Train], axis=1)


scaler = StandardScaler()
X_Val = scaler.fit(val_data[X_labels])
X_Val = scaler.transform(val_data[X_labels])

X_Val = pd.DataFrame(X_Val, columns=['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
                                     't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
                                     'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
                                     'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'])

X_Val = pd.concat([val_data['imageID'], X_Val], axis=1)

X_Test = scaler.fit(test_data[X_labels])
X_Test = scaler.transform(test_data[X_labels])

X_Test = pd.DataFrame(X_Test, columns=['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
                                       't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
                                       'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
                                       'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'])

X_Test = pd.concat([test_data['imageID'], X_Test], axis=1)


Y_Train = training_labels[Y_labels]
Y_Val = val_labels[Y_labels]

# #Support Vector Machines
classifier_svm = svm.SVC(probability=True, gamma='scale',
                         cache_size=350, decision_function_shape='ovo', C=100.0)
classifier_svm.fit(X_Train[X_labels], Y_Train.values.ravel())

Y_Pred_svm = classifier_svm.predict(X_Val[X_labels])
accuracy = accuracy_score(Y_Val, Y_Pred_svm)
print('Accuracy: {:.2f}'.format(accuracy))
print(classification_report(Y_Val, Y_Pred_svm,
                            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9], zero_division=0))


Y_test_pred = classifier_svm.predict_proba(X_Test[X_labels])
Y_pred_frame = pd.DataFrame({'ID': X_Test['imageID'], 'MEL': Y_test_pred[:, 0], 'NV': Y_test_pred[:, 1], 'BCC': Y_test_pred[:, 2], 'AK': Y_test_pred[:, 3],
                             'BKL': Y_test_pred[:, 4], 'DF': Y_test_pred[:, 5], 'VASC': Y_test_pred[:, 6], 'SCC': Y_test_pred[:, 7]})
Y_pred_frame['UNK'] = 0
Y_pred_frame.to_csv(os.path.join(
    os.getcwd(), 'data/val_1_1_svm.csv'), index=None, header=True)
