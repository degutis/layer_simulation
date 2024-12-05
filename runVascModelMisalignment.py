import simulation as sim
import numpy as np
import pickle as pkl
import os
import vasc_model as vm
import plotResults
from scipy import stats


layers=3
beta=0.035
trials = 50
iterations=20

rho_values = [0.4]
CNR_change = [1]
rval = len(rho_values)
CNR_values = len(CNR_change)
propChange=[-0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2]

folder_path = '../derivatives/pipeline_files/' 
folders_layers = [f.name for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith(f'Layers{layers}_Beta{beta}_Trials{trials}_LayerOfInt')]

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
accuracy_new  = np.empty((layers, iterations, rval, CNR_values, len(propChange)))

for index, folder in enumerate(sorted_folders):
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            for it in range(iterations):
                pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'
                betaRange = [beta, beta*CNR_change[ib]]

                with open(pathName, 'rb') as f:
                    _, _, boldPattern1,boldPattern2 = pkl.load(f)

                
                for ip, prop in enumerate(propChange):
                    input_seed = [(it*iterations + ip)]
                    vox = sim.VoxelResponses(it,r,r, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                    X,y,_,_ = vox.oneDecodable_changeVascModel(boldPattern1, boldPattern2, prop)                  
                    X_deconvolved = vm.deconvolve(X) # added a deconvolution step
                    accuracy_new[:,it,i,ib,ip] = vox.runSVM_classifier_acrossLayers(X_deconvolved, y)

    accuracy_old = np.load(f'../derivatives/results/Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}.npy')
    accuracy_diff = accuracy_new - np.repeat(accuracy_old[..., np.newaxis], len(propChange), axis=4)
    np.save(f'../derivatives/results/Difference_Accuracy_VascChanges{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisProp{propChange}.npy', accuracy_diff)

    plotResults.plotChangeMisalignment(accuracy_diff, rho_values, CNR_change, propChange, name_dict2[index], f'VascModelChange{name_dict2[index]}')
    
    accuracy_layerSubtraction = accuracy_new - accuracy_new[index, :,:,:]
    t_stat, _ = stats.ttest_1samp(accuracy_layerSubtraction, 0, axis=1)
    plotResults.plotTstat(t_stat, rho_values, CNR_change, propChange, name_dict2[index], f'VascModelChange_Ttest{name_dict2[index]}')
