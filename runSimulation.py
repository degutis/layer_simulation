import simulation as sim
import vasc_model as vm
import plotResults
import createFolders as cf
import numpy as np

cf.createFolders()

# Define some parameters
iterations=30
layers = 3
rho_values = [0.4, 0.5, 0.6, 0.7, 0.8]
rval = len(rho_values)

numTrials_per_class = 50
beta = 0.035

accuracy_samePattern = np.empty((layers,iterations, rval))
accuracy_deep = np.empty((layers,iterations, rval))
accuracy_middle = np.empty((layers,iterations, rval))
accuracy_superficial = np.empty((layers,iterations, rval))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
        
        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, beta=beta)
        
        X,y = vox.samePatternAcrossColumn()
        accuracy_samePattern[:,it,i] = vox.runSVM_classifier_acrossLayers(X,y)

        X_0,y_0 = vox.diffPatternsAcrossColumn_oneDecodable(0)
        accuracy_deep[:,it,i] = vox.runSVM_classifier_acrossLayers(X_0, y_0)
        
        X_1,y_1 = vox.diffPatternsAcrossColumn_oneDecodable(1)
        accuracy_middle[:,it,i] = vox.runSVM_classifier_acrossLayers(X_1, y_1)
        
        X_2,y_2 = vox.diffPatternsAcrossColumn_oneDecodable(2)
        accuracy_superficial[:,it,i] = vox.runSVM_classifier_acrossLayers(X_2, y_2)

nSize = [1,len(rho_values)]
plotResults.plotViolin(accuracy_samePattern, rho_values, nSize, "SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_deep, rho_values, nSize, "Deep")
plotResults.plotViolin(accuracy_middle, rho_values, nSize, "Middle")
plotResults.plotViolin(accuracy_superficial, rho_values, nSize, "Superficial")
