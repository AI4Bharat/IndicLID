from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
file_path = '../result_threshold_0.8/confusion_matrix_test_dakshina_filter_roman.csv'

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

disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=column_names)
fig, ax = plt.subplots(figsize=(16,16))
# ax.set_xticklabels(xticklabels, rotation = 45, ha="right")
disp.plot(cmap='Blues', colorbar=False, ax = ax, xticks_rotation=45)

# plt.xticks(tick_marks, target_names, rotation=45)
# plt.show()
plt.savefig('../result_threshold_0.8/confusion_matrix_test_dakshina_filter_roman.png')