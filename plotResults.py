import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotViolin(accuracy, rho_values, CNR_change, title):

    numLayers = accuracy.shape[0]
    numIterations = accuracy.shape[1]
    numParams = accuracy.shape[2]
    numBetas = accuracy.shape[3]

    if numLayers==3:
        layer_names = ["Deep", "Middle", "Superficial"]
    elif numLayers==4:
        layer_names = ["Deep", "Middle Deep", "Middle Superficial", "Superficial"] 
    elif numLayers==2:
        layer_names = ["Deep", "Superficial"] 


    # Flatten the tensor into a long format
    accuracy_flat = accuracy.flatten()

    # Create indices for Layer, Iteration, rho_values (Param1), and CNR_change (Param2)
    layers = np.repeat(layer_names, numIterations * numParams * numBetas)
    iterations = np.tile(np.repeat(np.arange(1, numIterations + 1), numParams * numBetas), numLayers)
    rhos = np.tile(np.repeat(rho_values, numBetas), numLayers * numIterations)
    betas = np.tile(CNR_change, numLayers * numIterations * numParams)

    # Create the DataFrame
    df = pd.DataFrame({
        'Layer': layers,
        'Iteration': iterations,
        'Rho': rhos,       
        'CNR_change': betas,
        'Accuracy': accuracy_flat
    })

    # Initialize the FacetGrid with rows for Rho and columns for CNR_change
    g = sns.FacetGrid(df, row='Rho', col='CNR_change', margin_titles=True, height=4, aspect=1)

    # Map the violinplot function to the grid, with hue for Layer
    g.map(sns.violinplot, 'Layer', 'Accuracy', order=layer_names, palette="Set2", inner = "points")

    g.map(plt.axhline, y=0.5, linestyle='--', color='gray')

    g.set_axis_labels("Layer", "Accuracy")
    g.set_titles(row_template="Rho = {row_name}", col_template="CNR_change = {col_name}")
    g.set(ylim=(0, 1))
    
    g.add_legend()

    # Show the plot
    plt.tight_layout()
    g.savefig(f"../derivatives/results/{title}.png",format="png")


def plotLaminarResp(X1, X2, FigTitle):

    if X1.ndim ==4:
        X1 = np.mean(X1,3)
        X2 = np.mean(X2,3)

    layers = X1.shape[2]   

    fig, axs = plt.subplots(2, layers, figsize=(15, 10), sharex='col', sharey='row')
    fig.text(0.07, 0.7, 'Pattern 1', va='center', rotation='vertical', fontsize=14)
    fig.text(0.07, 0.3, 'Pattern 2', va='center', rotation='vertical', fontsize=14)
        
    cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), cax=cbar_ax)

    for i in range(layers):
        axs[0, i].imshow(X1[:,:,i], cmap='gray')
        axs[0, i].set_title(f'Layer {i+1}')

        axs[1, i].imshow(X2[:,:,i], cmap='gray')

    fig.savefig(f'../derivatives/laminarPattern/LaminarResponse_{FigTitle}.png',format="png")
    plt.close(fig)


def setup_df(accuracy, layer_names, rho_values, CNR_change, percent_change, iterations=None):
    
    accuracy_flat = accuracy.flatten()
    numLayers = accuracy.shape[0]
    numParams = accuracy.shape[-3]
    numBetas = accuracy.shape[-2]
    numPercent = accuracy.shape[-1]
    
    layers = np.repeat(layer_names, numParams * numBetas * numPercent * (iterations or 1))
    rhos = np.tile(np.repeat(rho_values, numBetas * numPercent), numLayers * (iterations or 1))
    betas = np.tile(np.repeat(CNR_change, numPercent), numLayers * numParams * (iterations or 1))
    percentages = np.tile(percent_change, numLayers * numParams * numBetas * (iterations or 1))

    df_data = {
        'Layer': layers,
        'Rho': rhos,
        'CNR_change': betas,
        'Percent_change': percentages,
        'Accuracy': accuracy_flat
    }
    
    if iterations:
        df_data['Iteration'] = np.tile(np.repeat(np.arange(1, iterations + 1), numParams * numBetas * numPercent), numLayers)
    
    return pd.DataFrame(df_data)

def plotGraph(df, title, y_label, hue_order, layer_of_interest=None):
    
    g = sns.FacetGrid(df, row='Rho', col='CNR_change', margin_titles=True, height=4, aspect=1)
    g.map_dataframe(sns.lineplot, x='Percent_change', y='Accuracy', hue='Layer', 
                    hue_order=hue_order, palette="Set2", errorbar=('se'), markers=True)
    g.set_axis_labels("Misalignment Percent", y_label)
    g.set_titles(row_template="Rho = {row_name}", col_template="CNR_change = {col_name}")
    g.map(plt.axhline, y=0, linestyle='--', color='gray')
    g.add_legend(title="Layer", bbox_to_anchor=(1, 0.5), loc='center left')
    g.fig.suptitle(f'Separable patterns present in the {layer_of_interest} Layer')
    plt.tight_layout()
    g.savefig(f"../derivatives/results/{title}.png", format="png")

def plotChangeMisalignment(accuracy, rho_values, CNR_change, percent_change, title):
    
    layer_names = ["Deep", "Middle", "Superficial"] if accuracy.shape[0] == 3 else ["Deep", "Middle Deep", "Middle Superficial", "Superficial"]
    df = setup_df(accuracy, layer_names, rho_values, CNR_change, percent_change, iterations=accuracy.shape[1])
    plotGraph(df, title, "Accuracy Difference", layer_names)

def plotTstat(accuracy, rho_values, CNR_change, percent_change, layer_of_interest, title):
    
    layer_names = [f'{name} - {layer_of_interest}' for name in ["Deep", "Middle Deep", "Middle Superficial", "Superficial"][:accuracy.shape[0]]]
    df = setup_df(accuracy, layer_names, rho_values, CNR_change, percent_change)
    plotGraph(df, title, "T-value (One-sample t-test)", layer_names, layer_of_interest)
