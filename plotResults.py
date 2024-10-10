import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotViolin(accuracy, rho_values, nSize, title):

    numLayers = accuracy.shape[0]
    numParams = accuracy.shape[2]
    numIterations = accuracy.shape[1]

    accuracy_flat = accuracy.transpose(0,2,1).reshape(numLayers * numParams * numIterations)
    layers = np.repeat(np.arange(1, numLayers+1), numParams * numIterations) 
    parameters = np.tile(np.repeat(rho_values, numIterations), numLayers)  
    accuracy_values = np.tile(np.arange(1, numIterations+1), numLayers * numParams)

    df = pd.DataFrame({
        'Layer': layers,
        'Rho': parameters,
        'Accuracy': accuracy_flat
    })

    palette = sns.color_palette("Set2", 3)  
    fig, axes = plt.subplots(nrows=nSize[0], ncols=nSize[1], figsize=(20, 6), sharey=True)
    for i, rho in enumerate(rho_values):
        ax = axes[i]        
        sns.violinplot(x='Layer', y='Accuracy', data=df[df['Rho'] == rho], 
                    ax=ax, hue='Layer', palette=palette, inner = "points", split=False, legend=False)
        ax.set_title(f'Rho = {rho}')
        ax.set_xlabel('Layer')
        if i == 0:
            ax.set_ylabel('Accuracy')
        else:
            ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f"../derivatives/results/{title}.png")
    plt.close(fig)


