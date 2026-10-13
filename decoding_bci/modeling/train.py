from pathlib import Path

import typer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from decoding_bci.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()

def _get_features_and_labels(features_path, labels_path):
    df_train = pd.read_csv(str(features_path))
    df_labels = pd.read_csv(str(labels_path), dtype = np.int32)
    features = df_train.to_numpy(dtype=np.float32)
    labels = df_labels.to_numpy(dtype=np.int32).flatten()
    return features, labels

def load_model():
    model = Sequential()
    model.add(Dense(256, activation = 'relu', input_shape = (143,)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(8, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy', 'precision', 'recall'])
    return model

def load_data(features_path, labels_path):
    df_train = pd.read_csv(str(features_path))
    df_labels = pd.read_csv(str(labels_path), dtype = np.int32)
    features = df_train.to_numpy(dtype=np.float32)
    labels = df_labels.to_numpy(dtype=np.int32).flatten()  # shape: (n_samples,)
    
    full_data = tf.data.Dataset.from_tensor_slices((features, labels))
    
    TRAINING_NUM = int(0.70 * len(full_data))
    VAL_NUM = int(0.20 * len(full_data))
    TEST_NUM = int(0.10 * len(full_data))
    training_data = full_data.take(TRAINING_NUM).batch(1).prefetch(tf.data.AUTOTUNE)
    validation_data = full_data.skip(TRAINING_NUM).batch(1).take(VAL_NUM).prefetch(tf.data.AUTOTUNE)

    return training_data, validation_data

def hidden_layer_representation(trained_model, train_data):
    model_for_plot = load_model()
    for i, layer in enumerate(model_for_plot.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    hidden_features = model_for_plot.predict(train_data)
    return hidden_features

def generate_embedding(trained_model, train_data, features_path, labels_path, fig_path, TRAINING_NUM = 110):
    hidden_features = hidden_layer_representation(trained_model, train_data)
    tsne = TSNE(n_components = 2)
    features, labels = _get_features_and_labels(features_path, labels_path)
    labels = labels[:TRAINING_NUM]
    X_tsne = tsne.fit_transform(hidden_features)

    colors = np.array([x for x in "b-g-r-c-m-y-k-purple-coral-lime".split("-")])
    for color in range(8):
        idxs = np.where(labels == color)
        plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], c = colors[color], label = color * 45)

    plt.title("Neural Network Embedding of Neuron Firing Data")
    plt.legend()
    plt.savefig(str(fig_path))
    plt.show()




@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "ReachDataTrain.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "ReachDataLabels.csv",
    model_path: Path = MODELS_DIR / "model.keras",
    figures_path: Path = FIGURES_DIR / "Neural Network Embeddings.png"
):
    print("Loading data and model..")
    training_data, validation_data = load_data(features_path, labels_path)
    model = load_model()
    
    print("Training model...")
    model.fit(training_data, validation_data = validation_data, epochs = 25)
    print("Modeling training complete. Generating embeddings...")
    generate_embedding(model, training_data, features_path, labels_path, figures_path)

    print(f"Saving to {str(model_path)}...")
    model.save(str(model_path))

    


if __name__ == "__main__":
    app()
