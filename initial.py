import numpy as np 
import pandas as pd

voxels = 100
numTrials_per_class = 40

sd_signal_class1 = 0.5
sd_signal_class2 = 0.5
sd_noise = 0.2


activity_row_class1 = np.random.normal(1, sd_signal_class1, (1, voxels))
activity_row_class2 = np.random.normal(1, sd_signal_class2, (1, voxels))

activity_matrix_class1 = np.tile(activity_row_class1, (numTrials_per_class, 1))
activity_matrix_class2 = np.tile(activity_row_class2, (numTrials_per_class, 1))

activity_matrix_combined = np.concatenate(activity_matrix_class1, activity_matrix_class2)
noise = np.random.normal(0, sd_noise, (numTrials_per_class*2, voxels))
activity_matrix_combined = activity_matrix_combined + noise

class1 = np.full((numTrials_per_class, 1), 0) # class 1
class2 = np.full((numTrials_per_class, 1), 1) # class 2

y = np.concatenate(class1, class2)
permutation_indices = np.random.permutation(activity_matrix_combined.shape[0])

activity_matrix_permuted = activity_matrix_combined[permutation_indices, :]
y_permuted = y[permutation_indices]


print(permutation_indices.shape)
print(y_permuted.shape)






