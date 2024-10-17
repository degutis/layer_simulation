import simulation as sim
import vasc_model as vm
import plotResults
import createFolders as cf
import numpy as np

cf.createFolders()

# Define some parameters
iterations=30
layers = 3
rho_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
rval = len(rho_values)

numTrials_per_class = 50
beta = 0.02

accuracy_samePattern = np.empty((layers,iterations, rval))
accuracy_diffPattern = np.empty((layers,iterations, rval))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
        
        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, beta=beta)
        
        X,y = vox.samePatternAcrossColumn()
        accuracy_samePattern[:,it,i] = vox.runSVM_classifier_acrossLayers(X,y)

        X_diff, y_diff = vox.diffPatternsAcrossColumn()
        accuracy_diffPattern[:,it,i] = vox.runSVM_classifier_acrossLayers(X_diff, y_diff)

nSize = [1,len(rho_values)]
plotResults.plotViolin(accuracy_samePattern, rho_values, nSize, "SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_diffPattern, rho_values, nSize, "DiffPatternAcrossLayers")
