import simulation as sim
import vasc_model as vm
import plotResults
import createFolders as cf
import numpy as np

cf.createFolders()

# Define some parameters
iterations=30
layers = 3
rho_values = [0.5, 0.6]
CNR_change = [0.1, 0.5]
rval = len(rho_values)
CNR_values = len(CNR_change)

numTrials_per_class = 50
betaRange = [0.035, 0.035*2]

accuracy_samePattern = np.empty((layers,iterations, rval, CNR_values))
accuracy_deep = np.empty((layers,iterations, rval, CNR_values))
accuracy_middle = np.empty((layers,iterations, rval, CNR_values))
accuracy_superficial = np.empty((layers,iterations, rval, CNR_values))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
        for ib,b in enumerate(CNR_change):
        
            vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, betaRange=betaRange)
            
            X,y, _, _ = vox.samePatternAcrossColumn()
            accuracy_samePattern[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X,y)

            X_0,y_0,_, _ = vox.diffPatternsAcrossColumn_oneDecodable(0)
            accuracy_deep[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_0, y_0)
            
            X_1,y_1,_, _ = vox.diffPatternsAcrossColumn_oneDecodable(1)
            accuracy_middle[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_1, y_1)
            
            X_2,y_2,_, _ = vox.diffPatternsAcrossColumn_oneDecodable(2)
            accuracy_superficial[:,it,i, ib] = vox.runSVM_classifier_acrossLayers(X_2, y_2)

plotResults.plotViolin(accuracy_samePattern, rho_values, CNR_change, "GRID_SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_deep, rho_values, CNR_change, "GRID_Deep")
plotResults.plotViolin(accuracy_middle, rho_values, CNR_change, "GRID_Middle")
plotResults.plotViolin(accuracy_superficial, rho_values, CNR_change, "GRID_Superficial")
