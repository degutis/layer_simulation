import simulation as sim
import plotResults
import createFolders as cf
import numpy as np
import sys
from pathlib import Path
import pickle as pkl

cf.createFolders()

# Define some parameters
iterations=20
layers = 3
depths = layers*3

rho_initial_values = np.arange(0.2, 0.7, 0.1)
rho_matrix = np.array([[np.round(start + i * 0.05, 3) for i in range(depths)] for start in rho_initial_values])
rho_matrix = rho_matrix[:, ::-1]
rval = rho_matrix.shape[0]

rMatrix1 = np.array([0.6, 0.7, 0.8, 0.9, 1])
rMatrix2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
rho_matrix1 = np.tile(rMatrix1.reshape(-1, 1), depths)
rho_matrix2 = np.tile(rMatrix2.reshape(-1, 1), depths)

CNR_change = [1, 2, 3, 4]
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

accuracy = np.empty((layers,iterations, rval, CNR_values))

pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}'
Path(pathName).mkdir(parents=True, exist_ok=True)

for it in range(iterations):
    for i in range(rval):
        for ib,b in enumerate(CNR_change):
            betaRange = [beta, beta*CNR_change[ib]]
 #           vox = sim.VoxelResponses(it,rho_matrix[i,:], rho_matrix[i,:], numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            vox = sim.VoxelResponses(it,rho_matrix1[i,:], rho_matrix2[i,:], numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            X,y,_, _ = vox.diffPatternsAcrossColumn()
            accuracy[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

#rho_values = rho_matrix[:,0]
rho_values = rMatrix1
plotResults.plotViolin(accuracy, rho_values, CNR_change, "Layers_GRID_DiffPatterns_DiffRhoSameAcrossLayer_04Diff")
