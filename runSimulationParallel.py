import simulation as sim
import plotResults
import createFolders as cf
import numpy as np
import sys
from pathlib import Path
import pickle as pkl
import stats

cf.createFolders()

# Define some parameters
layer_index = int(sys.argv[1])  # This defines the layers to decode (e.g., [9, 10, 11], etc.)
print(layer_index)
iterations=20
layers = 3
rho_values = [0.4] 
CNR_change = [1]
rval = len(rho_values)
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

if layers==3:
    layer_dict = {
        0: [0,1,2],
        1: [3,4,5],
        2: [6,7,8],
        3: [0,1,2,6,7,8]
    }

    name_dict = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        3: "Deep and Superficial"
    }

elif layers==4:
    
    layer_dict = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
        3: [9, 10, 11],
    }

    name_dict = {
        0: "Deep",
        1: "Middle Deep",
        2: "Middle Superficial",
        3: "Superficial",
    }


pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}_LayerOfInt{name_dict[layer_index]}'
Path(pathName).mkdir(parents=True, exist_ok=True)

accuracy_file = f'../derivatives/results/Accuracy_LayerResponse{str(layer_index)}_rho{str(rho_values)}_CNR{str(CNR_change)}.npy'

try:
    accuracy = np.load(accuracy_file)
    print(f"Loaded accuracy data from {accuracy_file}")


except FileNotFoundError:
    accuracy = np.empty((layers,iterations, rval, CNR_values))
    for it in range(iterations):
        for i,r in enumerate(rho_values):
            for ib,b in enumerate(CNR_change):
                betaRange = [beta, beta*CNR_change[ib]]
                nameFile = f'{pathName}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                try:
                    with open(nameFile, 'rb') as handle:
                        X, y = pkl.load(handle)

                except:
                    vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers, N_depth=layers*3)                       
                    X,y,_,_,boldPattern1, boldPattern2 = vox.diffPatternsAcrossColumn_oneDecodable(layer_dict[layer_index])

                    with open(nameFile, 'wb') as handle:
                        pkl.dump((X, y, boldPattern1, boldPattern2), handle)

                accuracy[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

    np.save(accuracy_file, accuracy)

stats.runThreeLayers(accuracy,f'GRID_{name_dict[layer_index]}.txt')
plotResults.plotViolin(accuracy, rho_values, CNR_change, f'GRID_{name_dict[layer_index]}')