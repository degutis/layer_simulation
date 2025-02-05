import pickle as pkl
import numpy as np
import os

import simulation as sim
import plotResults
import vasc_model as vm
import stats

layers=3
beta=0.035
trials = 50
iterations=20
layers = 3
rho_values = [0.4] 
CNR_change = [1]
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
        5: "LayerOfIntDeep and Superficial"
    }

    name_dict2 = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        5: "Deep and Superficial"
    }

sorted_folders = sorted(folders_layers, key=lambda x: next(i for i, suffix in name_dict.items() if x.endswith(suffix)))
X_new  = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values))

accuracy_new  = np.empty((layers, iterations, rval, CNR_values))
univarResponse_old  = np.empty((layers, iterations, rval, CNR_values))
univarResponse_new  = np.empty((layers, iterations, rval, CNR_values))

for index, folder in enumerate(sorted_folders):
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            for it in range(iterations):
                pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                with open(pathName, 'rb') as f:
                    X_loaded, y, _,_ = pkl.load(f)

                X_new[:,:,:,it,i, ib] = vm.deconvolve(X_loaded) 
                betaRange = [beta, beta*CNR_change[ib]]
                
                vox = sim.VoxelResponses(it,r,r, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                accuracy_new[:,it,i,ib] = vox.runSVM_classifier_acrossLayers(X_new[:,:,:,it,i,ib], y)
    
                univarResponse_old[:,it,i,ib] = np.mean(X_loaded, (0,1))
                univarResponse_new[:,it,i,ib] = np.mean(X_new[:,:,:,it,i, ib],(0,1))

    plotResults.plotViolin(accuracy_new, rho_values, CNR_change, f'Deconvolution_{name_dict2[index]}')
    plotResults.plotUnivar(univarResponse_old, univarResponse_new, rho_values, CNR_change, name_dict2[index], f'DeconvolutionUnivar_{name_dict2[index]}')
    stats.runThreeLayers(accuracy_new,f'Deconvolution_{name_dict2[index]}.txt')
