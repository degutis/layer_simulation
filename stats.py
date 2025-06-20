import scipy.stats as stats
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols


def runThreeLayers(accuracy,output_name):
    
    Path("../derivatives/stats").mkdir(parents=True, exist_ok=True)

    outputDir = f"../derivatives/stats/{output_name}.txt"
        
    anova_three = stats.f_oneway(accuracy[0,:],accuracy[1,:], accuracy[2,:])
    deep_middle = stats.ttest_rel(accuracy[0,:],accuracy[1,:])
    middle_sup = stats.ttest_rel(accuracy[1,:],accuracy[2,:])
    deep_sup = stats.ttest_rel(accuracy[0,:],accuracy[2,:])

    results = [
    ("anova_three", float(anova_three.statistic), float(anova_three.pvalue)),
    ("deep_middle", float(deep_middle.statistic), float(deep_middle.pvalue)),
    ("middle_sup", float(middle_sup.statistic), float(middle_sup.pvalue)),
    ("deep_sup", float(deep_sup.statistic), float(deep_sup.pvalue))]

    # Writing to a text file
    with open(outputDir,"w") as file:
        file.write("Test Name\Statistic\tP-Value\n")
        for result in results:
            file.write(f"{result[0]}\t{result[1]:.6f}\t{result[2]:.6e}\n")


def twoWayAnova(accuracy,output_name):

    Path("../derivatives/stats").mkdir(parents=True, exist_ok=True)

    outputDir = f"../derivatives/stats/{output_name}.txt"

    if accuracy.shape[3]==1:
        accuracy = accuracy[:,:,:,0]

    layers, subjects, rhos = accuracy.shape
    df_list = []

    for layer in range(layers):
        for participant in range(subjects):
            for size in range(rhos):
                df_list.append([layer, size, accuracy[layer, participant, size]])

    df = pd.DataFrame(df_list, columns=["Layer", "Size", "Accuracy"])

    model = ols("Accuracy ~ C(Layer) * C(Size)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

#    data = {
#        "Layer": np.repeat(np.arange(layers), subjects * rhos),
#        "Size": np.tile(np.repeat(np.arange(rhos), subjects), layers),
#        "Accuracy": accuracy.ravel()
#            "Size": np.repeat(np.arange(rhos), subjects * layers),
#            "Layer": np.tile(np.repeat(np.arange(layers), subjects), rhos),
#            "Accuracy": accuracy.ravel()
#            }
#    df = pd.DataFrame(data)
#    print(df.head(50))
    
    # Run two-way ANOVA with interaction
#    model = ols("Accuracy ~ C(Layer) * C(Size)", data=df).fit()
#    anova_table = sm.stats.anova_lm(model, typ=2)

    with open(outputDir, "w") as file:
        file.write("\t".join(["Row", "sum_sq", "df", "F", "PR(>F)"]) + "\n")
        file.write("-" * 50 + "\n")
        for index, row in anova_table.iterrows():
            file.write(f"{index}\t{row['sum_sq']}\t{row['df']}\t{row['F']}\t{row['PR(>F)']}\n")

    
    # Writing results to a text file
#    with open(outputDir, "w") as file:
#        file.write("Source\tSum Squares\tDf\tF-Statistic\tP-Value\n")
#        for row in anova_table.itertuples():
#            file.write(f"{row.Index}\t{row.sum_sq:.6f}\t{row.df}\t{row.F:.6f}\t{row['PR(>F)']:.6e}\n")