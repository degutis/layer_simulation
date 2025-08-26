import simulation as sim
import plotResults
import createFolders as cf
import vasc_model as vm
import stats

import numpy as np
import pickle as pkl
from pathlib import Path

# Define some parameters 3/4/5 for diff/same/same_psf
layer_index = 5 

iterations=20
layers = 3
depths = layers*3

cf.createFolders(layers)

rho_initial_values = np.arange(0.2, 0.7, 0.1)
rho_matrix = np.array([[np.round(start + i * 0.05, 3) for i in range(depths)] for start in rho_initial_values])
rho_matrix = rho_matrix[:, ::-1]
rval = rho_matrix.shape[0]

rMatrix1 = [0.4, 0.5, 0.6, 0.7]
rMatrix2 = [0.4, 0.5, 0.6, 0.7]
rval = len(rMatrix1)

CNR_change = [1]
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

if layers==3:
    name_dict = {
        3: "Different",
        4: "Same",
        5: "Same_PSF_equil"
    }


accuracy = np.empty((layers,iterations, rval, CNR_values))
accuracy_deconvolved = np.empty((layers,iterations, rval, CNR_values))

pathName = f'../derivatives/pipeline_files/Layers{layers}_Beta{beta}_Trials{numTrials_per_class}_LayerOfInt{name_dict[layer_index]}'
Path(pathName).mkdir(parents=True, exist_ok=True)

for it in range(iterations):
    for i,r in enumerate(rMatrix1):
        for ib,b in enumerate(CNR_change):
            
            betaRange = [beta, beta*CNR_change[ib]]
            nameFile = f'{pathName}/Iteration{it}_Rho{r}_CNR{b}.pickle'

            try:
                with open(nameFile, 'rb') as handle:
                    X, y = pkl.load(handle)

            except:
                if layer_index == 3: 
                    vox = sim.VoxelResponses(it,r, r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
                    X,y,_,_,boldPattern1,boldPattern2 = vox.diffPatternsAcrossColumn()
                
                elif layer_index == 4:
                    vox = sim.VoxelResponses(it,r, r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)                       
                    X,y,_,_,boldPattern1,boldPattern2 = vox.samePatternAcrossColumn()
                
                elif layer_index == 5:
                    vox = sim.VoxelResponses(it,r, r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers, fwhmRange = [0.83, 0.83])                       
                    X,y,_,_,boldPattern1,boldPattern2 = vox.samePatternAcrossColumn()

                accuracy[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

                with open(nameFile, 'wb') as handle:
                    pkl.dump((X, y, boldPattern1, boldPattern2), handle)

            accuracy[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X, y)

            X_deconvolved = vm.deconvolve(X) 
            accuracy_deconvolved[:,it, i, ib] = vox.runSVM_classifier_acrossLayers(X_deconvolved, y)

stats.twoWayAnova(accuracy,f'{name_dict[layer_index]}_twoWayANOVA.txt')
stats.twoWayAnova(accuracy_deconvolved,f'{name_dict[layer_index]}_Deconvolved_twoWayANOVA.txt')

for i,r in enumerate(rMatrix1):
    rho_current = [r]
    for ib,b in enumerate(CNR_change):
        CNR_current = [b]
        accuracy_file = f'../derivatives/results_layers{layers}/Accuracy_LayerResponse{str(layer_index)}_rho{str(rho_current)}_CNR{str(CNR_current)}.npy'
        np.save(accuracy_file, accuracy[:,:,i,ib])
        stats.runThreeLayers(accuracy[:,:,i,ib],f'Null_{name_dict[layer_index]}_CNR_{CNR_current}_rho_{rho_current}.txt')

        accuracy_file_dec = f'../derivatives/results_layers{layers}/Deconvolution_Accuracy_LayerResponse{str(layer_index)}_rho{str(rho_current)}_CNR{str(CNR_current)}.npy'
        np.save(accuracy_file_dec, accuracy_deconvolved[:,:,i,ib])
        stats.runThreeLayers(accuracy_deconvolved[:,:,i,ib],f'{name_dict[layer_index]}_Deconvolved_CNR_{CNR_current}_rho_{rho_current}.txt')



rho_values = rMatrix1

if layer_index == 3: 
    plotResults.plotViolin(accuracy, rho_values, CNR_change, "Layers_GRID_DiffPatternAcrossLayers")
    plotResults.plotViolin(accuracy_deconvolved, rho_values, CNR_change, "Layers_GRID_DiffPatternAcrossLayers_deconvolved")
elif layer_index == 4: 
    plotResults.plotViolin(accuracy, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers")
    plotResults.plotViolin(accuracy_deconvolved, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers_deconvolved")
elif layer_index == 5: 
    plotResults.plotViolin(accuracy, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers_noLam2Ddiff")
    plotResults.plotViolin(accuracy_deconvolved, rho_values, CNR_change, "Layers_GRID_SamePatternAcrossLayers_deconvolved_noLam2Ddiff")