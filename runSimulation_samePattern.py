import simulation as sim
import plotResults
import createFolders as cf
import numpy as np
import sys
from pathlib import Path
import pickle as pkl

cf.createFolders()

# Define some parameters
iterations=30
layers = 2
depths = 12

rho_initial_values = np.arange(0.2, 0.7, 0.1)
rho_matrix = np.array([[np.round(start + i * 0.05, 3) for i in range(depths)] for start in rho_initial_values])
rho_matrix = rho_matrix[:, ::-1]
rval = rho_matrix.shape[0]

CNR_change = [1.5,2,2.5,3]
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
            vox = sim.VoxelResponses(it,rho_matrix[i,:], rho_matrix[i,:], numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            X,y,_, _ = vox.diffPatternsAcrossColumn()
            accuracy[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

rho_values = rho_matrix[:,0]
plotResults.plotViolin(accuracy, rho_values, CNR_change, "2Layers_GRID_DiffPatterns_rhoChangeAcrossLayers")
