import simulation as sim
import vasc_model as vm
import plotResults

import numpy as np


iterations=30
layers = 3
rho_values = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
rval = len(rho_values)

numTrials_per_class = 50
beta = 0.02

accuracy_samePattern = np.empty((layers,iterations, rval))
accuracy_onePatternDeep = np.empty((layers,iterations,rval))
accuracy_onePatternMiddle = np.empty((layers,iterations,rval))
accuracy_onePatternSup = np.empty((layers,iterations,rval))

for it in range(iterations):
    print(it)
    for i,r in enumerate(rho_values):
    
        vox = sim.VoxelResponses(it,r,r, numTrials_per_class=numTrials_per_class, beta=beta)
        vox.plotPattern(f"VoxelResponses_Iteration_{it}_Rho_{r}")

        drainedResp = vox.calculateDrainingEffect_samePatternAcrossLayers(layers=layers)    
        accuracy_samePattern[:,it,i] = vox.runSVM_classifier_acrossLayers(drainedResp.outputMatrix)   
    
        drainingVeinDeep = vox.calculateDrainingEffect_patternOnlyInSingleLayer(0, it+10, layers=layers)
        accuracy_onePatternDeep[:,it,i] = vox.runSVM_classifier_acrossLayers(drainingVeinDeep.outputMatrix)   
    
        drainingVeinMiddle = vox.calculateDrainingEffect_patternOnlyInSingleLayer(1, it+20, layers=layers)
        accuracy_onePatternMiddle[:,it,i] = vox.runSVM_classifier_acrossLayers(drainingVeinMiddle.outputMatrix)   
    
        drainingVeinSup = vox.calculateDrainingEffect_patternOnlyInSingleLayer(2, it+30, layers=layers)
        accuracy_onePatternSup[:,it,i] = vox.runSVM_classifier_acrossLayers(drainingVeinSup.outputMatrix)   

nSize = [1,7]
plotResults.plotViolin(accuracy_samePattern, rho_values, nSize, "SamePatternAcrossLayers")
plotResults.plotViolin(accuracy_onePatternDeep, rho_values, nSize, "OnePatternDeep")
plotResults.plotViolin(accuracy_onePatternMiddle, rho_values, nSize, "OnePatternMiddle")
plotResults.plotViolin(accuracy_onePatternSup, rho_values, nSize, "OnePatternSup")
