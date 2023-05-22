from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
file_path = '../result/train_combine_confusion_matrix_word_overlap.csv'

# file = open('', 'r')
# file.read().split()

# my_data = genfromtxt(file_path, delimiter=',')
# print(my_data)

df = pd.read_csv(file_path, header=None)
np_array = df.values

column_names = np_array[0][1:]
print(len(column_names))

cf_matrix = np_array[1:, 1:]
print(cf_matrix.shape)

cf_matrix = cf_matrix.astype(np.int32)

for i in range(len(column_names)):
    cf_matrix[i][i] = 0 




predictions = []

cf_matrix = cf_matrix.tolist()
for row in range(len(cf_matrix)):    
    for col in range(len(cf_matrix)):
        for i in range(cf_matrix[row][col]):
            predictions.append([column_names[row], column_names[col]])

predictions = np.array(predictions)
y_test = predictions[:,0]
y_pred = predictions[:,1]

# without values
# disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, include_values=False, normalize='true')
# fig, ax = plt.subplots(figsize=(16,16))

# disp.plot(cmap='Greens', colorbar=False, ax = ax, xticks_rotation=45, include_values=False)
# plt.savefig('../result/train_combine_confusion_matrix_word_overlap_without_values.png')


# with values
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, include_values=True)
fig, ax = plt.subplots(figsize=(16,16))

disp.plot(cmap='Greens', colorbar=False, ax = ax, xticks_rotation=45, include_values=True)
plt.savefig('../result/train_combine_confusion_matrix_word_overlap_with_values.png')












# disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=column_names)
# fig, ax = plt.subplots(figsize=(16,16))
# # ax.set_xticklabels(xticklabels, rotation = 45, ha="right")
# disp.plot(cmap='Greens', colorbar=False, ax = ax, xticks_rotation=45)
# disp.ax_.set(xlabel='Languages', ylabel='Languages')

# # plt.xticks(tick_marks, target_names, rotation=45)
# # plt.show()
# plt.savefig('../result/train_combine_confusion_matrix_word_overlap.png')