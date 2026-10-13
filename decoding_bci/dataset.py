from pathlib import Path

import typer
import numpy as np
import pandas as pd
from scipy.io import loadmat

from decoding_bci.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "ReachData.mat",
    output_path: Path = PROCESSED_DATA_DIR,
):
    print("Processing dataset...")
    data = loadmat(str(input_path))
    dirs = data["direction"]
    neuron_firing = data["R"]

    #transposing matrix so that each row is one trial with each entry in the row as a neuron
    neuron_firing = np.transpose(neuron_firing)

    #0-indexing labels in dirs
    dirs = dirs - 1

    #exporting neuron firing and direction data to csv
    df = pd.DataFrame(neuron_firing)
    df.to_csv(str(output_path / "ReachDataTrain.csv"), index = False)
    df = pd.DataFrame(dirs)
    df.to_csv(str(output_path / "ReachDataLabels.csv"), index = False)
    
    print("Processing dataset complete.")


if __name__ == "__main__":
    app()
