from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

file_path = '../result_threshold_0.6/confusion_matrix_test_combine_roman.csv'

# file = open('', 'r')
# file.read().split()

# my_data = genfromtxt(file_path, delimiter=',')
# print(my_data)

df = pd.read_csv(file_path, header=None)
np_array = df.values

column_names = np_array[0][1:21]

predictions = []

cf_matrix = np_array[1:21, 1:21]
cf_matrix = cf_matrix.astype(int)
cf_matrix = cf_matrix.tolist()
for row in range(len(cf_matrix)):    
    for col in range(len(cf_matrix)):
        for i in range(cf_matrix[row][col]):
            predictions.append([column_names[row], column_names[col]])

predictions = np.array(predictions)
y_test = predictions[:,0]
y_pred = predictions[:,1]

# without values
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, include_values=False, normalize='true')
fig, ax = plt.subplots(figsize=(16,16))

disp.plot(cmap='OrRd', colorbar=False, ax = ax, xticks_rotation=45, include_values=False)
plt.savefig('../result_threshold_0.6/confusion_matrix_test_combine_roman_without_values.png')


# with values
# disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, include_values=True)
# fig, ax = plt.subplots(figsize=(16,16))

# disp.plot(cmap='OrRd', colorbar=False, ax = ax, xticks_rotation=45, include_values=True)
# plt.savefig('../result_threshold_0.6/confusion_matrix_test_combine_roman_with_values.png')

















# print(cf_matrix)

# cf_matrix = np_array[1:21, 1:21]
# print(cf_matrix.shape)

# cf_matrix = cf_matrix.astype(np.int32)

# # with values
# disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=column_names)
# fig, ax = plt.subplots(figsize=(16,16))

# disp.plot(cmap='Blues', colorbar=False, ax = ax, xticks_rotation=45)
# plt.savefig('../result_threshold_0.6/confusion_matrix_test_combine_roman_without_en_other_with_values.png')

# ConfusionMatrixDisplay.from_estimator(
# ...     clf, X_test, y_test)