from pathlib import Path

import typer
import pandas as pd
import tensorflow as tf
import numpy as np

from decoding_bci.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def load_test_data(features_path):
    df = pd.read_csv(str(features_path))
    dataset = tf.data.Dataset.from_tensor_slices((df))

    TEST_NUM = int(0.10 * len(dataset))
    test_data = dataset.skip(len(dataset) - TEST_NUM).take(TEST_NUM).batch(1)
    return test_data

def load_test_labels(labels_path):
    df = pd.read_csv(str(labels_path))
    labels = df.to_numpy(dtype=np.int32).flatten()

    #only get last few labels
    TEST_NUM = int(0.10 * len(labels))
    test_labels = labels[-TEST_NUM:len(labels)]
    return test_labels

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "ReachDataTrain.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "ReachDataLabels.csv",
    model_path: Path = MODELS_DIR / "model.keras",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    print("Loading data and model...")
    test_data = load_test_data(features_path)
    test_labels = load_test_labels(labels_path)
    model = tf.keras.models.load_model(str(model_path))

    print("Running predictions...")
    predictions = model.predict(test_data)
    predictions = np.argmax(predictions, axis = 1)

    print(f"Saving predictions to {str(predictions_path)}...")
    df = pd.DataFrame({"predictions": predictions, "actual": test_labels})
    df.to_csv(str(predictions_path), index = False)
    



if __name__ == "__main__":
    app()
