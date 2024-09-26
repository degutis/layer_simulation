import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import columnsfmri as cf
import vasc_model as vm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class VoxelResponses:
    def __init__(self, seed, rho_c1, rho_c2, N=512, L = 24, deltaRelative=0.5, fwhm = 1.02, beta = 0.035, samplingVox=1, numTrials_per_class=60):
        
        # 64:3 ratio between N and L.

        self.N = N   
        self.L = L     
        self.deltaRelative = deltaRelative
        self.fwhm = fwhm
        self.beta = beta
        self.w = samplingVox # =1

        self.seed = seed
        self.numTrials_per_class = numTrials_per_class

        self.activity_row_class1 = self.__generateInitialColumnarPattern__(seed, rho_c1)
        self.activity_row_class2 = self.__generateInitialColumnarPattern__(seed*2, rho_c2)

        activity_matrix_class1 = np.tile(self.activity_row_class1, (self.numTrials_per_class, 1))
        activity_matrix_class2 = np.tile(self.activity_row_class2, (self.numTrials_per_class, 1))
        activity_matrix_combined = np.concatenate((activity_matrix_class1, activity_matrix_class2), axis=0)
        
        #self.activity_matrix_combined_with_noise = self.__generateNoiseMatrix__(activity_matrix_combined)

        self.activity_matrix_combined_with_noise = activity_matrix_combined

        class1 = np.full((numTrials_per_class, 1), 0) # class 1
        class2 = np.full((numTrials_per_class, 1), 1) # class 2

        y = np.concatenate((class1, class2), axis=0)
        self.permutation_indices = np.random.permutation(self.activity_matrix_combined_with_noise.shape[0])

        self.activity_matrix_permuted = self.activity_matrix_combined_with_noise[self.permutation_indices, :]
        self.y_permuted = y[self.permutation_indices]

    def __generateInitialColumnarPattern__(self, seed, rho):
        
        # N, L, seed = 100, 24, 1
        # N - size of grid. L - size of patch in mm
        
        # rho,deltaRelative = 1, 0.5  
        # Rho is the main pattern frequency, delta specifies the amount of irregularity
        
        # fwhm = 1.02; deltaRelative = 0.035  
        # spatial BOLD response with a FWHM of 1.02 mm (7T GE), and a corresponding single condition average response amplitude of 3.5%.       

        # w = 1 Size of voxel

        sim = cf.simulation(self.N, self.L, seed)
        gwn = sim.gwnoise();
        columnPattern, _ = sim.columnPattern(rho,self.deltaRelative,gwn)
        boldPattern, _, _ = sim.bold(self.fwhm,self.beta,columnPattern)
        mriPattern = sim.mri(self.w, boldPattern)
        return mriPattern.reshape(-1)
        

    def __generateNoiseMatrix__(self, mriPattern, sliceThickness = 2.5, TR = 2, nT = 1000, differentialFlag = True, noiseType="7T"):    
        
        V = self.w**2*sliceThickness
        SNR = 1/cf.noiseModel(V,TR,nT,differentialFlag,noiseType=noiseType)
   
        return mriPattern + (1/SNR) * np.random.randn(*mriPattern.shape)
   
    def calculateTSNR(self):

        mean_signal = np.mean(self.activity_matrix_permuted, axis=-1)
        std_noise = np.std(self.activity_matrix_permuted, axis=-1)
        tsnr = np.divide(mean_signal, std_noise, where=std_noise!=0)
        print(f"Mean tSNR across the brain: {np.mean(tsnr)}")


    def combineLayersIntoMatrix(self):
        return 

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
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy * 100:.2f}%")       







