import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pkl
import os
from pathlib import Path

import columnsfmri as cf
import vasc_model as vm

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



class VoxelResponses:
    def __init__(self, seed, rho_c1, rho_c2, N=320, L = 16, N_depth=12, layers=4, deltaRelative=0.5, betaRange = [0.035, 0.035*2], samplingVox=1, numTrials_per_class=50, fwhmRange = [0.83, 1.78]):
        
        # 64:3 ratio between N and L.

        self.N = N   
        self.L = L   
        self.N_depth = N_depth
        self.layers = layers
        
        self.deltaRelative = deltaRelative
        self.w = samplingVox
        
        self.fwhmRange = fwhmRange
        step_fwhm = (self.fwhmRange[1] - self.fwhmRange[0]) / (self.N_depth - 1)
        self.fwhm_layers = [self.fwhmRange[0] + step_fwhm * i for i in range(self.N_depth)]
        
        self.betaRange = betaRange
        step_beta = (self.betaRange[1] - self.betaRange[0]) / (self.N_depth - 1)
        self.beta_layers = [self.betaRange[0] + step_beta * i for i in range(self.N_depth)]

        self.rho_c1 = rho_c1
        self.rho_c2 = rho_c2

        self.seed = seed
        np.random.seed(self.seed)
        self.numTrials_per_class = numTrials_per_class

    def samePatternAcrossColumn(self):

        activity_row_class1, boldPattern_class1  = self.__generateLaminarColumnarPattern__(self.seed, self.rho_c1)
        activity_row_class2, boldPattern_class2  = self.__generateLaminarColumnarPattern__(self.seed+13086, self.rho_c2)

        activity_matrix_permuted, y_permuted = self.__createTrialMatrix__(activity_row_class1, activity_row_class2)

        return activity_matrix_permuted, y_permuted, boldPattern_class1, boldPattern_class2

    def diffPatternsAcrossColumn(self):

        activity_row_class1, boldPattern_class1  = self.__generateLaminarPatternsDifferent__(self.seed, self.rho_c1)
        activity_row_class2, boldPattern_class2  = self.__generateLaminarPatternsDifferent__(self.seed+13086, self.rho_c2)

        activity_matrix_permuted, y_permuted = self.__createTrialMatrix__(activity_row_class1, activity_row_class2)

        return activity_matrix_permuted, y_permuted, boldPattern_class1, boldPattern_class2

    def diffPatternsAcrossColumn_oneDecodable(self, layer_of_interest):

        activity_row_class1, boldPattern_class1  = self.__generateLaminarPatternSingleLayer__(self.seed, self.rho_c1, layer_of_interest)
        activity_row_class2, boldPattern_class2  = self.__generateLaminarPatternSingleLayer__(self.seed+13086, self.rho_c2, layer_of_interest)

        activity_matrix_permuted, y_permuted = self.__createTrialMatrix__(activity_row_class1, activity_row_class2)

        return activity_matrix_permuted, y_permuted, boldPattern_class1, boldPattern_class2


    def __generateLaminarPatternsDifferent__(self, seed, rho):

        columnPattern = np.empty((self.N, self.N, self.N_depth))
        boldPattern = np.empty((self.N, self.N, self.N_depth))
        mriPattern = np.empty((self.L, self.L, self.layers))

        for la in range(self.N_depth):
            sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, seed)
            gwn = sim.gwnoise()
            columnPattern[:,:, la], _ = sim.columnPattern(rho,self.deltaRelative,gwn)

        for l in range(self.N_depth):
            boldPattern[:, :, l], _, _ = sim.bold(self.fwhm_layers[l], self.beta_layers[l],columnPattern[:,:,l])

        drainedSignal = vm.vascModel(boldPattern.transpose((2,1,0)), layers=self.N_depth)

        mriPattern = sim.mri(self.w, drainedSignal.outputMatrix.transpose(1,2,0))

        return mriPattern.reshape(mriPattern.shape[0]*mriPattern.shape[1],mriPattern.shape[2]), drainedSignal.outputMatrix.transpose(1,2,0)        


    def __generateLaminarColumnarPattern__(self, seed, rho):

        sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, seed)
        gwn = sim.gwnoise()
        columnPattern, _ = sim.columnPattern(rho,self.deltaRelative,gwn)
        boldPattern = np.empty((self.N, self.N, self.N_depth))
        mriPattern = np.empty((self.L, self.L, self.layers))
                
        for l in range(self.N_depth):
            boldPattern[:, :, l], _, _ = sim.bold(self.fwhm_layers[l], self.beta_layers[l],columnPattern)

        drainedSignal = vm.vascModel(boldPattern.transpose((2,1,0)), layers=self.N_depth)

        mriPattern = sim.mri(self.w, drainedSignal.outputMatrix.transpose(1,2,0))

        return mriPattern.reshape(mriPattern.shape[0]*mriPattern.shape[1],mriPattern.shape[2]), drainedSignal.outputMatrix.transpose(1,2,0)       


    def __generateLaminarPatternSingleLayer__(self, seed, rho, layer_of_interest):
        
        boldPattern = np.empty((self.N, self.N, self.N_depth, self.numTrials_per_class))
        columnPattern = np.empty((self.N, self.N, self.N_depth, self.numTrials_per_class))
        drainedSignal_output = np.empty((self.N, self.N, self.N_depth, self.numTrials_per_class))
        mriPattern = np.empty((self.L, self.L, self.layers, self.numTrials_per_class))

        layer_range = range(self.N_depth)
        
        pathName = f'../derivatives/pipeline_files/N{self.N}_L{self.L}_Ndepth{self.N_depth}_Layers{self.layers}_BetaRange{str(self.betaRange)}_RhoValues_{rho}'

        Path(pathName).mkdir(parents=True, exist_ok=True)

        for tr in range(self.numTrials_per_class):
            input_seeds = [(tr*self.N_depth + la + (seed+1)) if la not in layer_of_interest else seed for la in layer_range]
            nameFile = f'{pathName}/BoldPattern_Seed_{str(seed)}_Trial_{str(tr)}.pickle'
            if not os.path.exists(nameFile):
                for la in layer_range:
                    sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, input_seeds[la])
                    gwn = sim.gwnoise()          
                    columnPattern[:,:, la, tr], _ = sim.columnPattern(rho, self.deltaRelative, gwn)
                    boldPattern[:, :, la, tr], _, _ = sim.bold(self.fwhm_layers[la], self.beta_layers[la], columnPattern[:,:, la, tr])

                with open(nameFile, 'wb') as handle:
                    pkl.dump((sim, boldPattern[:,:,:,tr], layer_of_interest), handle)

            else: 
                with open(nameFile, 'rb') as f:
                    sim, boldPattern_loaded, layer_of_interest_loaded = pkl.load(f)
                
                boldPattern_loaded[:, :, layer_of_interest_loaded], boldPattern_loaded[:,:, layer_of_interest] = boldPattern_loaded[:, :, layer_of_interest].copy(), boldPattern_loaded[:, :, layer_of_interest_loaded].copy()
                boldPattern[:,:,:,tr] = boldPattern_loaded
                    
            drainedSignal = vm.vascModel(boldPattern[:, :, :, tr].transpose((2, 1, 0)), layers=self.N_depth)
            drainedSignal_output[:, :, :, tr] = drainedSignal.outputMatrix.transpose(1, 2, 0)

            mriPattern[:, :, :, tr] = sim.mri(self.w, drainedSignal_output[:, :, :, tr])

        mriPattern_reshaped = mriPattern.reshape(mriPattern.shape[0] * mriPattern.shape[1], mriPattern.shape[2], mriPattern.shape[3])
        
        return mriPattern_reshaped.transpose(2,0,1), drainedSignal_output       


    def __createTrialMatrix__(self, activity_row_class1, activity_row_class2):

        if activity_row_class1.ndim==2:
            activity_matrix_class1 = np.tile(activity_row_class1, (self.numTrials_per_class, 1, 1))
            activity_matrix_class2 = np.tile(activity_row_class2, (self.numTrials_per_class, 1, 1))
            activity_matrix_combined = np.concatenate((activity_matrix_class1, activity_matrix_class2), axis=0)
        else:
            activity_matrix_combined = np.concatenate((activity_row_class1, activity_row_class2), axis=0)
        
        activity_matrix_combined_with_noise = self.__generateNoiseMatrix__(activity_matrix_combined)

        class1 = np.full((self.numTrials_per_class, 1), 0) # class 1
        class2 = np.full((self.numTrials_per_class, 1), 1) # class 2

        y = np.concatenate((class1, class2), axis=0)
        permutation_indices = np.random.permutation(activity_matrix_combined_with_noise.shape[0])

        activity_matrix_permuted = activity_matrix_combined_with_noise[permutation_indices, :, :]
        y_permuted = y[permutation_indices]

        return activity_matrix_permuted, y_permuted

    def __generateNoiseMatrix__(self, mriPattern, sliceThickness = 1, TR = 2, nT = 1, differentialFlag = False, noiseType="7T"):    
        
        # nt - number acquisitions - otherwise have autocorrelation. 

        V = self.w**2*sliceThickness
        self.SNR = 1/cf.noiseModel(V,TR,nT,differentialFlag,noiseType=noiseType)

        rg = np.random.RandomState(self.seed+56)

        return mriPattern + (1/self.SNR) * rg.randn(*mriPattern.shape)
   

    #def calculateTSNR(self):

        #mean_signal = np.mean(self.activity_matrix_permuted, axis=-1)
        #std_noise = np.std(self.activity_matrix_permuted, axis=-1)
        #tsnr = np.divide(mean_signal, std_noise, where=std_noise!=0)
        #print(f"Mean tSNR across the brain: {np.mean(tsnr)}")

    """
    def plotPattern(self,FigTitle, save=True, show=False):
        
        fig = plt.figure(figsize=(15, 15))

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
        
        if save==True:
            fig.savefig(f'../derivatives/pattern_simulation/WorkflowPattern_{FigTitle}.png',format="png")
        if show==True:
            plt.show()
        
        plt.close(fig)
    """

    def runSVM_classifier_acrossLayers(self, layer_responses, y_permuted, n_splits=5):
        """
        Runs an SVM classifier for each layer using cross-validation and a standard scaler

        Parameters:
        layer_responses : float : a layer x voxel x trial matrix
        seed: int : random seed
        n_splits: int: number of times to split the data

        Returns:
        scores : meaned cross validated accuracy scores for each layer. [Layers, ] vector (0 index - deepest layer)
        """
   
        if layer_responses.shape[0]>20:
            layer_responses = layer_responses.transpose(2,1,0)

        self.n_splits = n_splits

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())               
        ])

        layers = layer_responses.shape[0]
        y = np.ravel(y_permuted)
        scores = np.empty(layers, dtype=float)
        
        for layer in range(layers):

            X = layer_responses[layer,:,:].transpose((1,0))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = self.seed+layer)
            scores[layer] = np.mean(cross_val_score(pipeline, X, y, cv=cv))

        return scores


    def calculateDrainingEffect_patternOnlyInSingleLayer(self, signalLayer, seed, layers=10):
        """
        Repeats the same underlying columnar pattern across all layers and computes
        the draining vein effect using vasc_model

        Parameters:
        layers : int : number of layers to estimate the draining vein effect on
    
        Returns:
        class : parameters and main output matrix from vasc_model
        """

        rg = np.random.RandomState(seed)
        noise = (1/self.SNR) * rg.randn(self.activity_matrix_permuted.shape[0],self.activity_matrix_permuted.shape[1],layers-1)
        activity_matrix_permuted_extra_dim = self.activity_matrix_permuted[:,:,np.newaxis]
        combinedLayersMatrix = np.concatenate((activity_matrix_permuted_extra_dim, noise), axis=2)
        combinedLayersMatrix[:,:,[0, signalLayer]] = combinedLayersMatrix[:,:,[signalLayer, 0]]
        combinedLayersMatrix = combinedLayersMatrix.transpose((2,1,0))

        return vm.vascModel(combinedLayersMatrix, layers=layers)


