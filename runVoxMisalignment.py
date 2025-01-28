import simulation as sim
import numpy as np
import pickle as pkl
import os
import plotResults
from scipy import stats

percent_change=[1,5,10,15,20,30,40]

layers=3
beta=0.035
trials = 50
iterations=20

rho_values = [0.4]
CNR_change = [1]
rval = len(rho_values)
CNR_values = len(CNR_change)
voxels = 256

folder_path = '../derivatives/pipeline_files/' 
folders_layers = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith(f'Layers{layers}_Beta{beta}_Trials{trials}_LayerOfInt')]

if layers==3:

    name_dict = {
        0: "LayerOfIntDeep",
        1: "LayerOfIntMiddle",
        2: "LayerOfIntSuperficial",
        3: "LayerOfIntDifferent",
        4: "LayerOfIntSame"
    }

    name_dict2 = {
        0: "Deep",
        1: "Middle",
        2: "Superficial",
        3: "Different",
        4: "Same"
    }

sorted_folders = sorted(folders_layers, key=lambda x: next(i for i, suffix in name_dict.items() if x.endswith(suffix)))
print(sorted_folders)

X = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values, len(sorted_folders)))
X_new  = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values, len(percent_change)))

accuracy_new  = np.empty((layers, iterations, rval, CNR_values, len(percent_change)))

for index, folder in enumerate(sorted_folders):
    print(index)
    print(folder)
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            for it in range(iterations):
                pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                with open(pathName, 'rb') as f:
                    X_loaded, y, _,_ = pkl.load(f)

                X[:,:,:,it,i, ib,index] = X_loaded
                betaRange = [beta, beta*CNR_change[ib]]
                
                for ip, percent in enumerate(percent_change):
                    input_seed = [(it*iterations + percent)]
                    vox = sim.VoxelResponses(it,r,r, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                    X_new[:,:,:,it,i,ib,ip] = sim.missegmentationVox(X_loaded, percent, input_seed)                   
                    accuracy_new[:,it,i,ib,ip] = vox.runSVM_classifier_acrossLayers(X_new[:,:,:,it,i,ib,ip], y)

    accuracy_old = np.load(f'../derivatives/results/Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}.npy')
    
    if accuracy_old.ndim < 4:
        accuracy_old = np.reshape(accuracy_old, accuracy_old.shape + (1,) * (4 - accuracy_old.ndim))
    
    accuracy_diff = accuracy_new - np.repeat(accuracy_old[..., np.newaxis], len(percent_change), axis=4)
    
    if index==3:
        accuracy_layerSubtraction = accuracy_new - accuracy_new[0, :,:,:] #subtract deep layer in diff
        accuracy_layerSubtraction[0,:,:,:] = accuracy_new[2,:,:,:] - accuracy_new[1,:,:,:]

    elif index==4:
        accuracy_layerSubtraction = accuracy_new - accuracy_new[1, :,:,:] #subtract middle layer in same 
        accuracy_layerSubtraction[1,:,:,:] = accuracy_new[0,:,:,:] - accuracy_new[2,:,:,:]
    else:
        accuracy_layerSubtraction = accuracy_new - accuracy_new[index, :,:,:]

    t_stat, _ = stats.ttest_1samp(accuracy_layerSubtraction, 0, axis=1)

    np.save(f'../derivatives/results/Difference_Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisPerc{percent_change}.npy', accuracy_diff)
    np.save(f'../derivatives/results/Difference_Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisPerc{percent_change}.npy', t_stat)  
    
    plotResults.plotChangeMisalignment(accuracy_diff, rho_values, CNR_change, percent_change, name_dict2[index], f'MisalignmentChange{name_dict2[index]}')
    plotResults.plotTstat(t_stat, rho_values, CNR_change, percent_change, name_dict2[index], f'MisalignmentLayersChange{name_dict2[index]}')





