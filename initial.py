import numpy as np 
import pandas as pd


class VoxelResponses:
    def __init__(self, voxels=100, numTrials_per_class=40, sd_signal_class1 = 0.5, sd_signal_class2 = 0.5, sd_noise= 0.2):
        
        self.voxels = voxels
        self.numTrials_per_class = numTrials_per_class

        self.sd_signal_class1 = sd_signal_class1
        self.sd_signal_class2 = sd_signal_class2
        self.sd_noise = sd_noise

        activity_row_class1 = np.random.normal(1, self.sd_signal_class1, (1, voxels))
        activity_row_class2 = np.random.normal(1, self.sd_signal_class2, (1, voxels))

        activity_matrix_class1 = np.tile(activity_row_class1, (self.numTrials_per_class, 1))
        activity_matrix_class2 = np.tile(activity_row_class2, (self.numTrials_per_class, 1))

        activity_matrix_combined = np.concatenate((activity_matrix_class1, activity_matrix_class2), axis=0)
        self.noise = np.random.normal(0, sd_noise, (numTrials_per_class*2, voxels))
        self.activity_matrix_combined_with_noise = activity_matrix_combined + self.noise

        class1 = np.full((numTrials_per_class, 1), 0) # class 1
        class2 = np.full((numTrials_per_class, 1), 1) # class 2

        y = np.concatenate((class1, class2), axis=0)
        self.permutation_indices = np.random.permutation(self.activity_matrix_combined_with_noise.shape[0])

        self.activity_matrix_permuted = activity_matrix_combined[self.permutation_indices, :]
        self.y_permuted = y[self.permutation_indices]


obj1 = VoxelResponses()

print(obj1.y_permuted.shape)








