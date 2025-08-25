from pathlib import Path

def createFolders(layers):

    Path(f'../derivatives/results_layers{layers}').mkdir(parents=True, exist_ok=True)
    Path(f'../derivatives/pipeline_files').mkdir(parents=True, exist_ok=True)
