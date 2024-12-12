import simulation as sim
import plotResults
import createFolders as cf
import numpy as np
import sys
from pathlib import Path
import pickle as pkl
import vasc_model as vm

cf.createFolders()

# Define some parameters
iterations=20
layers = 3
depths = layers*3

rho_initial_values = np.arange(0.2, 0.7, 0.1)
rho_matrix = np.array([[np.round(start + i * 0.05, 3) for i in range(depths)] for start in rho_initial_values])
rho_matrix = rho_matrix[:, ::-1]
rval = rho_matrix.shape[0]

rMatrix1 = [0.4, 0.5, 0.6, 0.7]
rMatrix2 = [0.4, 0.5, 0.6, 0.7]
#rMatrix2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
#rho_matrix1 = np.tile(rMatrix1.reshape(-1, 1), depths)
#rho_matrix2 = np.tile(rMatrix2.reshape(-1, 1), depths)
rval = len(rMatrix1)

CNR_change = [1]
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

accuracy = np.empty((layers,iterations, rval, CNR_values))
accuracy_deconvolved = np.empty((layers,iterations, rval, CNR_values))

pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}'
Path(pathName).mkdir(parents=True, exist_ok=True)

for it in range(iterations):
    #for i in range(rval):
    for i,r in enumerate(rMatrix1):
        for ib,b in enumerate(CNR_change):
            betaRange = [beta, beta*CNR_change[ib]]
 #           vox = sim.VoxelResponses(it,rho_matrix[i,:], rho_matrix[i,:], numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
 #           vox = sim.VoxelResponses(it,rho_matrix1[i,:], rho_matrix2[i,:], numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            vox = sim.VoxelResponses(it,r, r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            
            X,y,_, _ = vox.samePatternAcrossColumn()
            accuracy[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

            X_deconvolved = vm.deconvolve(X) 
            accuracy_deconvolved[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X_deconvolved, y)
    
#rho_values = rho_matrix[:,0]
rho_values = rMatrix1
plotResults.plotViolin(accuracy, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_deconvolved, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers_deconvolved")
