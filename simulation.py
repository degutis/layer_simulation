import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import columnsfmri as cf
import vasc_model as vm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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
        np.random.seed(self.seed)
        self.numTrials_per_class = numTrials_per_class

        self.activity_row_class1, self.columnPattern1 = self.__generateInitialColumnarPattern__(seed, rho_c1)
        self.activity_row_class2, self.columnPattern2 = self.__generateInitialColumnarPattern__(seed+13086, rho_c2)

        activity_matrix_class1 = np.tile(self.activity_row_class1, (self.numTrials_per_class, 1))
        activity_matrix_class2 = np.tile(self.activity_row_class2, (self.numTrials_per_class, 1))
        activity_matrix_combined = np.concatenate((activity_matrix_class1, activity_matrix_class2), axis=0)
        
        self.activity_matrix_combined_with_noise = self.__generateNoiseMatrix__(activity_matrix_combined)

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
        gwn = sim.gwnoise()
        columnPattern, _ = sim.columnPattern(rho,self.deltaRelative,gwn)
        boldPattern, _, _ = sim.bold(self.fwhm,self.beta,columnPattern)
        mriPattern = sim.mri(self.w, boldPattern)
        return mriPattern.reshape(-1), columnPattern
        

    def __generateNoiseMatrix__(self, mriPattern, sliceThickness = 2.5, TR = 2, nT = 1000, differentialFlag = True, noiseType="7T"):    
        
        V = self.w**2*sliceThickness
        SNR = 1/cf.noiseModel(V,TR,nT,differentialFlag,noiseType=noiseType)

        rg = np.random.RandomState(self.seed+56)

        return mriPattern + (1/SNR) * rg.randn(*mriPattern.shape)
   

    def calculateTSNR(self):

        mean_signal = np.mean(self.activity_matrix_permuted, axis=-1)
        std_noise = np.std(self.activity_matrix_permuted, axis=-1)
        tsnr = np.divide(mean_signal, std_noise, where=std_noise!=0)
        print(f"Mean tSNR across the brain: {np.mean(tsnr)}")


    def plotPattern(self):
        
        plt.figure(figsize=(20, 20))

        ax1 = plt.subplot(6,6,(1,2))
        ax2 = plt.subplot(6,6,(3))
        ax3 = plt.subplot(6,6,(13,14))
        ax4 = plt.subplot(6,6,(15))
        ax5 = plt.subplot(6,6,(10,11))

        color1 = 'green'
        color2 = 'blue'

        ax1.imshow(self.columnPattern1, aspect='equal', cmap='gray', interpolation='none')
        ax1.set_title('Columnar pattern #1')
        ax1.set_ylabel('Grid points')
        ax1.set_xlabel('Grid points')

        ax3.imshow(self.columnPattern2, aspect='equal', cmap='gray', interpolation='none')
        ax3.set_title('Columnar pattern #2')
        ax3.set_ylabel('Grid points')
        ax3.set_xlabel('Grid points')

        ax2.plot(self.activity_row_class1, color = color1)
        ax2.set_title('Voxel activation #1')
        ax2.set_ylabel('Signal')
        ax2.set_xlabel('Voxels')

        ax4.plot(self.activity_row_class2, color = color2)
        ax4.set_title('Voxel activation #2')
        ax4.set_ylabel('Signal')
        ax4.set_xlabel('Voxels')

        ax5.imshow(self.activity_matrix_permuted, aspect='auto', cmap='gray', interpolation='none')
        for i in range(self.activity_matrix_permuted.shape[0]):
            rect_color = color1 if self.y_permuted[i] == 0 else color2
            rect = patches.Rectangle((0, i-0.5), self.activity_matrix_permuted.shape[1], 1, linewidth=0.5, edgecolor=rect_color, facecolor='none')
            ax5.add_patch(rect)       
        ax5.set_title(f"Synthetic fMRI data for all trials)")
        ax5.set_ylabel('Trials')
        ax5.set_xlabel('Voxels')

        plt.tight_layout()
        plt.savefig('../derivatives/pattern_simulation/pattern.png')


    def runSVM_classifier(self, test_size=0.2):
        
        np.random.seed(self.seed)
        self.test_size = test_size
        print(self.seed)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.activity_matrix_permuted, np.ravel(self.y_permuted), test_size = self.test_size, random_state=self.seed)
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        svm_model = SVC(kernel='linear')  
        svm_model.fit(self.X_train, self.y_train)
        self.y_pred = svm_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {self.accuracy * 100:.2f}%")       

    def calculateDrainingEffect_samePatternAcrossLayers(self, layers=10):

        layers_samePattern = np.repeat(self.activity_matrix_permuted[:,:,np.newaxis], layers, axis=2).transpose((2,1,0))
        return vm.vascModel(layers_samePattern, layers=layers)



