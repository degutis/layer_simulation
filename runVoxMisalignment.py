import simulation as sim
import numpy as np
import pickle as pkl
import os
import plotResults
from scipy import stats

percent_change=[1,5,10,15,20,30,40]

layers=4
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
X = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values, len(sorted_folders)))
X_new  = np.empty((trials*2, voxels, layers,iterations, rval, CNR_values, len(percent_change)))

accuracy_new  = np.empty((layers, iterations, rval, CNR_values, len(percent_change)))

for index, folder in enumerate(sorted_folders):
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            for it in range(iterations):
                pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'

                with open(pathName, 'rb') as f:
                    X_loaded, y = pkl.load(f)

                X[:,:,:,it,i, ib,index] = X_loaded
                betaRange = [beta, beta*CNR_change[ib]]
                
                for ip, percent in enumerate(percent_change):
                    input_seed = [(it*iterations + percent)]
                    vox = sim.VoxelResponses(it,r,r, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                    X_new[:,:,:,it,i,ib,ip] = sim.missegmentationVox(X_loaded, percent, input_seed)                   
                    accuracy_new[:,it,i,ib,ip] = vox.runSVM_classifier_acrossLayers(X_new[:,:,:,it,i,ib,ip], y)
    
    accuracy_old = np.load(f'../derivatives/results/Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}.npy')
    accuracy_diff = accuracy_new - np.repeat(accuracy_old[..., np.newaxis], len(percent_change), axis=4)
    np.save(f'../derivatives/results/Difference_Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisPerc{percent_change}.npy', accuracy_diff)

    plotResults.plotChangeMisalignment(accuracy_diff, rho_values, CNR_change, percent_change, name_dict2[index], f'MisalignmentChange{name_dict2[index]}')
    
    accuracy_layerSubtraction = accuracy_new - accuracy_new[index, :,:,:]
    t_stat, _ = stats.ttest_1samp(accuracy_layerSubtraction, 0, axis=1)
    plotResults.plotTstat(t_stat, rho_values, CNR_change, percent_change, name_dict2[index], f'MisalignmentLayersChange{name_dict2[index]}')





