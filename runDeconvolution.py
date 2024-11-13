import pickle as pkl
import numpy as np
import os

import simulation as sim
import plotResults
import vasc_model as vm

layers=3
beta=0.035
trials = 50
iterations=15

rho_values = [0.4, 0.5, 0.6]
CNR_change = [1, 2, 3]
rval = len(rho_values)
CNR_values = len(CNR_change)
voxels = 256

folder_path = '../derivatives/pipeline_files/' 
folders_layers = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith(f'Layers{layers}_Beta{beta}_Trials{trials}_LayerOfInt')]
print(folders_layers)

if layers==3:

    name_dict = {
        0: "LayerOfIntDeep",
        1: "LayerOfIntMiddle",
        2: "LayerOfIntSuperficial",
    }

    name_dict2 = {
        0: "Deep",
        1: "Middle",
        2: "Superficial"
    }

elif layers==4:

    name_dict = {
        0: "LayerOfIntDeep",
        1: "LayerOfIntMiddleDeep",
        2: "LayerOfIntMiddleSuperficial",
        3: "LayerOfIntSuperficial",
    }

    name_dict2 = {
        0: "Deep",
        1: "MiddleDeep",
        2: "MiddleSuperficial",
        3: "Superficial"
    }


sorted_folders = sorted(folders_layers, key=lambda x: next(i for i, suffix in name_dict.items() if x.endswith(suffix)))
X_new  = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values))

accuracy_new  = np.empty((layers, iterations, rval, CNR_values))

for index, folder in enumerate(sorted_folders):
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            for it in range(iterations):
                pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                with open(pathName, 'rb') as f:
                    X_loaded, y = pkl.load(f)

                #X[:,:,:,it,i, ib,index] = X_loaded
                X_new[:,:,:,it,i, ib] = vm.deconvolve(X_loaded) 
                betaRange = [beta, beta*CNR_change[ib]]
                
                vox = sim.VoxelResponses(it,r,r, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                accuracy_new[:,it,i,ib] = vox.runSVM_classifier_acrossLayers(X_new[:,:,:,it,i,ib], y)
    
    plotResults.plotViolin(accuracy_new, rho_values, CNR_change, f'Deconvolution_{name_dict2[index]}')
   




