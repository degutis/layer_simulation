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
propChange=[-0.5,-0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2,0.5]

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
accuracy_new  = np.empty((layers, iterations, rval, CNR_values, len(propChange)))

for index, folder in enumerate(sorted_folders):
    try:
        accuracy_diff_file = f'../derivatives/results/Difference_Accuracy_VascChanges{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisProp{propChange}.npy'
        tstat_file = f'../derivatives/results/Tstat_{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisProp{propChange}.npy'

        try:
            accuracy_diff = np.load(accuracy_diff_file)
            t_stat = np.load(tstat_file)
            print(f"Loaded accuracy data from {accuracy_diff_file}")
            print(f"Loaded accuracy data from {tstat_file}")
        
        except FileNotFoundError:
            print(index)
            print(folder)
            for i,r in enumerate(rho_values):
                for ib,b in enumerate(CNR_change):
                    for it in range(iterations):
                        pathName = f'{folder_path}/{sorted_folders[index]}/Iteration{it}_Rho{r}_CNR{b}.pickle'
                        betaRange = [beta, beta*CNR_change[ib]]

                        with open(pathName, 'rb') as f:
                            _, _, boldPattern1,boldPattern2 = pkl.load(f)
                        
                        for ip, prop in enumerate(propChange):
                            input_seed = [(it*iterations + ip)]
                            vox = sim.VoxelResponses(it,r,r+0.4, numTrials_per_class=trials, betaRange=betaRange, layers=layers)                       
                            X,y,_,_ = vox.oneDecodable_changeVascModel(boldPattern1, boldPattern2, prop)                  
                            X_deconvolved = vm.deconvolve(X) # added a deconvolution step
                            accuracy_new[:,it,i,ib,ip] = vox.runSVM_classifier_acrossLayers(X_deconvolved, y)

            accuracy_old = np.load(f'../derivatives/results/Accuracy_LayerResponse{index}_rho{(rho_values)}_CNR{str(CNR_change)}.npy')
            if accuracy_old.ndim < 4:
                accuracy_old = np.reshape(accuracy_old, accuracy_old.shape + (1,) * (4 - accuracy_old.ndim))
        
            accuracy_diff = accuracy_new - np.repeat(accuracy_old[..., np.newaxis], len(propChange), axis=4)  

            if index==3:
                accuracy_layerSubtraction = accuracy_new - accuracy_new[0, :,:,:] #subtract deep layer in diff
                accuracy_layerSubtraction[0,:,:,:] = accuracy_new[2,:,:,:] - accuracy_new[1,:,:,:]

            elif index==4:
                accuracy_layerSubtraction = accuracy_new - accuracy_new[1, :,:,:] #subtract middle layer in same 
                accuracy_layerSubtraction[1,:,:,:] = accuracy_new[0,:,:,:] - accuracy_new[2,:,:,:]
            else:
                accuracy_layerSubtraction = accuracy_new - accuracy_new[index, :,:,:]

            t_stat, _ = stats.ttest_1samp(accuracy_layerSubtraction, 0, axis=1)
            
            np.save(f'../derivatives/results/Difference_Accuracy_VascChanges{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisProp{propChange}.npy', accuracy_diff)
            np.save(f'../derivatives/results/Tstat_{index}_rho{(rho_values)}_CNR{str(CNR_change)}_MisProp{propChange}.npy', t_stat)

        plotResults.plotChangeMisalignment(accuracy_diff, rho_values, CNR_change, propChange, name_dict2[index], f'VascModelChange{name_dict2[index]}_rho{(rho_values)}')   
        plotResults.plotTstat(t_stat, rho_values, CNR_change, propChange, name_dict2[index], f'VascModelChange_Ttest{name_dict2[index]}_rho{(rho_values)}')
    except:
        continue