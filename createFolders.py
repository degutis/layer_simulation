from pathlib import Path

def createFolders():

    Path("../derivatives/results").mkdir(parents=True, exist_ok=True)
    Path("../derivatives/laminarPattern").mkdir(parents=True, exist_ok=True)
    Path("../derivatives/pattern_simulation").mkdir(parents=True, exist_ok=True)
