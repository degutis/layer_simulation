import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class VoxelResponses:
    def __init__(self, seed, voxels=100, numTrials_per_class=40, sd_signal_class1 = 10, sd_signal_class2 = 10, sd_noise= 20):
        
        self.seed = seed
        np.random.seed(self.seed)

        self.voxels, self.numTrials_per_class = voxels, numTrials_per_class

        self.sd_signal_class1, self.sd_signal_class2 = sd_signal_class1, sd_signal_class2 # not actually used now
        self.sd_noise = sd_noise

        self.baseline_signal = np.random.normal(100, 6, size=(self.numTrials_per_class*2, self.voxels))

        mean1 = np.ones(self.voxels) * 10
        cov1 = np.identity(self.voxels)  # might want to introduce some spatial dependencies within the pattern
        activity_row_class1 = np.random.multivariate_normal(mean1, cov1) 

        mean2 = np.ones(self.voxels) * 10
        cov2 = np.identity(self.voxels)  # might want to introduce some spatial dependencies within the pattern
        activity_row_class2 = np.random.multivariate_normal(mean1, cov1) 

        #activity_row_class1 = np.random.normal(0.1, self.sd_signal_class1, (1, voxels))
        #activity_row_class2 = np.random.normal(0.1, self.sd_signal_class2, (1, voxels))

        activity_matrix_class1 = np.tile(activity_row_class1, (self.numTrials_per_class, 1))
        activity_matrix_class2 = np.tile(activity_row_class2, (self.numTrials_per_class, 1))

        activity_matrix_combined = np.concatenate((activity_matrix_class1, activity_matrix_class2), axis=0)
        
        self.noise = np.random.normal(0, self.sd_noise, size=(numTrials_per_class*2, self.voxels,))  # Standard deviation = 20

        self.activity_matrix_combined_with_noise = activity_matrix_combined + self.noise + self.baseline_signal

        class1 = np.full((numTrials_per_class, 1), 0) # class 1
        class2 = np.full((numTrials_per_class, 1), 1) # class 2

        y = np.concatenate((class1, class2), axis=0)
        self.permutation_indices = np.random.permutation(self.activity_matrix_combined_with_noise.shape[0])

        self.activity_matrix_permuted = self.activity_matrix_combined_with_noise[self.permutation_indices, :]
        self.y_permuted = y[self.permutation_indices]

        ###
        mean_signal = np.mean(self.activity_matrix_permuted, axis=-1)
        std_noise = np.std(self.activity_matrix_permuted, axis=-1)
        tsnr = np.divide(mean_signal, std_noise, where=std_noise!=0)
        print(f"Mean tSNR across the brain: {np.mean(tsnr)}")

    def plotPattern(self):
        
        plt.imshow(self.activity_matrix_combined_with_noise, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label="Signal Intensity")
        plt.title(f"Synthetic fMRI Data (1D Space with {self.voxels} Voxels)")
        plt.xlabel("Time")
        plt.ylabel("Voxel (Space)")
        plt.savefig('../derivatives/pattern_simulation/pattern.png')

    def runSVM_classifier(self, test_size=0.2):

        self.test_size = test_size
        X_train, X_test, y_train, y_test = train_test_split(self.activity_matrix_permuted, np.ravel(self.y_permuted), test_size = self.test_size, random_state=self.seed)
        svm_model = SVC(kernel='linear')  
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")       


vox1 = VoxelResponses(seed=1, sd_noise=1)

vox1.plotPattern()





