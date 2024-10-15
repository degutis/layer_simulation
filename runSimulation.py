import simulation as sim
import vasc_model as vm
import plotResults
import createFolders as cf
import numpy as np
cf.createFolders()

# Define some parameters
iterations=30
layers = 3
rho_values = [0.5, 0.6, 0.7, 0.8, 0.9]
rval = len(rho_values)

numTrials_per_class = 50
beta = 0.07

accuracy_samePattern = np.empty((layers,iterations, rval))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
        
        FigTitle = f"_{str(it)}_Rho_{str(r).replace('.','')}"

        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, beta=beta)

        accuracy_samePattern[:,it,i] = vox.runSVM_classifier_acrossLayers(vox.activity_matrix_permuted)  

nSize = [1,len(rho_values)]
plotResults.plotViolin(accuracy_samePattern, rho_values, nSize, "SamePatternAcrossLayers")
