import numpy as np 

import columnsfmri as cf
import vasc_model as vm

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
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

        seed_list = [seed, seed+1, seed+2, seed+3] # three separate 
        seed_list = np.repeat(seed_list,3)

        if type(rho) == float:
            rho = np.repeat(rho,self.N_depth)

        for la in range(self.N_depth):
            sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, seed_list[la])
            gwn = sim.gwnoise()
            columnPattern[:,:, la], _ = sim.columnPattern(rho[la],self.deltaRelative,gwn)
            boldPattern[:, :, la], _, _ = sim.bold(self.fwhm_layers[la], self.beta_layers[la],columnPattern[:,:,la])

        drainedSignal = vm.vascModel(boldPattern.transpose((2,1,0)), layers=self.N_depth)

        mriPattern = sim.mri(self.w, drainedSignal.outputMatrix.transpose(1,2,0))

        return mriPattern.reshape(mriPattern.shape[0]*mriPattern.shape[1],mriPattern.shape[2]), drainedSignal.outputMatrix.transpose(1,2,0)        


    def __generateLaminarColumnarPattern__(self, seed, rho):

        sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, seed)
        gwn = sim.gwnoise()

        if type(rho) == int:
            columnPattern, _ = sim.columnPattern(rho,self.deltaRelative,gwn)
            boldPattern = np.empty((self.N, self.N, self.N_depth))
            mriPattern = np.empty((self.L, self.L, self.layers))    
            for l in range(self.N_depth):
                boldPattern[:, :, l], _, _ = sim.bold(self.fwhm_layers[l], self.beta_layers[l],columnPattern)

        else:
            for l in range(self.N_depth):
                columnPattern, _ = sim.columnPattern(rho[l],self.deltaRelative,gwn)
                boldPattern = np.empty((self.N, self.N, self.N_depth))
                mriPattern = np.empty((self.L, self.L, self.layers))    
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
        
        for tr in range(self.numTrials_per_class):
            input_seeds = [(tr*self.N_depth + la + (seed+1)) if la not in layer_of_interest else seed for la in layer_range]
            for la in layer_range:
                sim = cf.simulation(self.N, self.L, self.N_depth, self.layers, input_seeds[la])
                gwn = sim.gwnoise()          
                columnPattern[:,:, la, tr], _ = sim.columnPattern(rho, self.deltaRelative, gwn)
                boldPattern[:, :, la, tr], _, _ = sim.bold(self.fwhm_layers[la], self.beta_layers[la], columnPattern[:,:, la, tr])
                    
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
    

def missegmentationVox(X, percent, seed):

    np.random.seed(seed)
    layers = X.shape[2]
    num_indices = int(X.shape[1] * (percent / 100))
    X_new = X.copy()

    for l1 in range(layers-1):
        selected_indices = np.random.choice(X.shape[1], size=num_indices, replace=False)
        X_new[:,selected_indices,l1], X_new[:,selected_indices,l1+1] = X[:,selected_indices,l1+1], X[:,selected_indices,l1]

    return X_new