from pathlib import Path

def createFolders():

    Path("../derivatives/results").mkdir(parents=True, exist_ok=True)
    Path("../derivatives/pipeline_files").mkdir(parents=True, exist_ok=True)
