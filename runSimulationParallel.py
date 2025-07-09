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
# layer_index = 1
print(layer_index)
iterations=10
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
        5: [0,1,2,6,7,8]
    }

    name_dict = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        5: "Deep and Superficial"
    }

elif layers==6:
    
    layer_dict = {
        0: [0, 1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 10, 11],
        2: [12, 13, 14, 15, 16, 17],
        10: [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
        11: [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    }

    name_dict = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        8: "Deep2 and Superficial2",
        9: "Deep2 and Superficial2 Same PSF",
        10: "Deep and Superficial",
        11: "Deep2 and Superficial2 Same PSF"
    }

elif layers==9:
    
    layer_dict = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17],
        2: [18, 19, 20, 21, 22, 23, 24, 25, 26],
        8: [3,4,5, 21, 22, 23],
        9: [3,4,5, 21, 22, 23],
        10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        11: [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    }

    name_dict = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        8: "Deep2 and Superficial2",
        9: "Deep2 and Superficial2 Same PSF",
        10: "Deep and Superficial",
        11: "Deep2 and Superficial2 Same PSF"
    }


pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}_LayerOfInt{name_dict[layer_index]}'
Path(pathName).mkdir(parents=True, exist_ok=True)

accuracy_file = f'../derivatives/results/Accuracy_LayerResponse{str(layer_index)}_rho{str(rho_values)}_CNR{str(CNR_change)}.npy'
cnr_uni_file = f'../derivatives/results/CNRUni_LayerResponse{str(layer_index)}_rho{str(rho_values)}_CNR{str(CNR_change)}.npy'
cnr_multi_file = f'../derivatives/results/CNRMulti_LayerResponse{str(layer_index)}_rho{str(rho_values)}_CNR{str(CNR_change)}.npy'

try:
    accuracy = np.load(accuracy_file)
    cnr_uni = np.load(cnr_uni_file)
    cnr_multi = np.load(cnr_multi_file)
    print(f"Loaded accuracy data from {accuracy_file}")

except FileNotFoundError:
    accuracy = np.empty((layers,iterations, rval, CNR_values))
    cnr_uni = np.empty((layers,iterations, rval, CNR_values))
    cnr_multi = np.empty((layers,iterations, rval, CNR_values))

    for it in range(iterations):
        for i,r in enumerate(rho_values):
            for ib,b in enumerate(CNR_change):
                betaRange = [beta, beta*CNR_change[ib]]
                nameFile = f'{pathName}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                try:
                    with open(nameFile, 'rb') as handle:
                        X, y = pkl.load(handle)

                except:
                    
                    if layer_index==9 or layer_index==11:
                        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers, N_depth=layers*3, fwhmRange = [0.83, 0.83])                       
                    else:
                        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers, N_depth=layers*3)                       
                    
                    X,y,_,_,boldPattern1, boldPattern2 = vox.diffPatternsAcrossColumn_oneDecodable(layer_dict[layer_index])

                    with open(nameFile, 'wb') as handle:
                        pkl.dump((X, y, boldPattern1, boldPattern2), handle)
                
                cnr_uni[:,it,i, ib], cnr_multi[:,it,i, ib] = vox.runFullCNR(X,y)
                accuracy[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

    np.save(accuracy_file, accuracy)
    np.save(cnr_uni_file, cnr_uni)
    np.save(cnr_multi_file, cnr_multi)

stats.runThreeLayers(accuracy,f'GRID_{name_dict[layer_index]}_rho{str(rho_values)}.txt')
plotResults.plotViolin(accuracy, rho_values, CNR_change, f'GRID_{name_dict[layer_index]}_rho{str(rho_values)}')
plotResults.plotCNR(cnr_uni, rho_values, CNR_change, f'CNRuni_{name_dict[layer_index]}_rho{str(rho_values)}')
plotResults.plotCNR(cnr_multi, rho_values, CNR_change, f'CNRmulti_{name_dict[layer_index]}_rho{str(rho_values)}')