import simulation as sim
import plotResults
import createFolders as cf
import numpy as np
import sys
from pathlib import Path
import pickle as pkl

cf.createFolders()

# Define some parameters
layer_index = int(sys.argv[1])  # This defines the layers to decode (e.g., [9, 10, 11], etc.)
iterations=30
layers = 4
rho_values = [0.5, 0.6]
CNR_change = [2, 3]
rval = len(rho_values)
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

accuracy = np.empty((layers,iterations, rval, CNR_values))

layer_dict = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11]
}

name_dict = {
    0: "Deep",
    1: "Middle_Deep",
    2: "Middle_Sup",
    3: "Superficial"
}

pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}_LayerOfInt{name_dict[layer_index]}'
Path(pathName).mkdir(parents=True, exist_ok=True)

for it in range(iterations):
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            betaRange = [beta, beta*CNR_change[ib]]
            vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
            X,y,_, _ = vox.diffPatternsAcrossColumn_oneDecodable(layer_dict[layer_index])

            nameFile = f'{pathName}/Iteration{it}_Rho{r}_CNR{b}.pickle'
            with open(nameFile, 'wb') as handle:
                pkl.dump((X, y), handle)

            accuracy[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

np.save(f'../derivatives/results/Accuracy_LayerResponse{str(layer_index)}_rho{str(rho_values)}_CNR{str(CNR_values)}.npy', accuracy)
plotResults.plotViolin(accuracy, rho_values, CNR_change, f'GRID_{name_dict[layer_index]}')