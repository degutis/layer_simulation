import simulation as sim
import plotResults
import createFolders as cf
import numpy as np

cf.createFolders()

# Define some parameters
iterations=30
layers = 4
rho_values = [0.5, 0.6]
CNR_change = [2, 3]
rval = len(rho_values)
CNR_values = len(CNR_change)

beta = 0.035
numTrials_per_class = 50

accuracy_samePattern = np.empty((layers,iterations, rval, CNR_values))
accuracy_deep = np.empty((layers,iterations, rval, CNR_values))
accuracy_middle_deep = np.empty((layers,iterations, rval, CNR_values))
accuracy_middle_superficial = np.empty((layers,iterations, rval, CNR_values))
accuracy_superficial = np.empty((layers,iterations, rval, CNR_values))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
            
            betaRange = [beta, beta*CNR_change[ib]]

            vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange, layers=layers)
            
            X,y, _, _ = vox.samePatternAcrossColumn()
            accuracy_samePattern[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X,y)

            X_0,y_0,_, _ = vox.diffPatternsAcrossColumn_oneDecodable([0,1,2])
            accuracy_deep[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_0, y_0)
            
            X_1,y_1,_, _ = vox.diffPatternsAcrossColumn_oneDecodable([3,4,5])
            accuracy_middle_deep[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_1, y_1)
            
            X_2,y_2,_, _ = vox.diffPatternsAcrossColumn_oneDecodable([6,7,8])
            accuracy_middle_superficial[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_2, y_2)

            X_3,y_3,_, _ = vox.diffPatternsAcrossColumn_oneDecodable([9,10,11])
            accuracy_superficial[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_3, y_3)


plotResults.plotViolin(accuracy_samePattern, rho_values, CNR_change, "GRID_SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_deep, rho_values, CNR_change, "GRID_Deep")
plotResults.plotViolin(accuracy_middle_deep, rho_values, CNR_change, "GRID_Middle_Deep")
plotResults.plotViolin(accuracy_middle_superficial, rho_values, CNR_change, "GRID_Middle_Superficial")
plotResults.plotViolin(accuracy_superficial, rho_values, CNR_change, "GRID_Superficial")
