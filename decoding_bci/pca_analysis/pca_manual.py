from pathlib import Path

import typer
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

from decoding_bci.config import RAW_DATA_DIR, FIGURES_DIR

app = typer.Typer()

def center_data(neuron_firing):
    mean = np.mean(neuron_firing, axis = 1, keepdims = True)
    centered_neuron_firing = neuron_firing - mean
    return centered_neuron_firing

def compute_covariance(centered_neuron_firing):
    covariance = np.cov(centered_neuron_firing, rowvar = True)
    return covariance

def compute_eigen(covariance):
    eigenvalues, eigenvectors = eig(covariance)

    #making sure to order values from greatest to least
    indx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indx]
    eigenvectors = eigenvectors[:, indx]
    return eigenvalues, eigenvectors

def compute_two_principal_component_variance(eigenvalues, eigenvectors):
    variance = np.cumsum(eigenvalues) / sum(eigenvalues)
    return variance[1]

def visualize(centered_neuron_firing, directions, eigenvectors, figures_path):
    projected_data = eigenvectors.T @ centered_neuron_firing

    colors = np.array([x for x in "b-g-r-c-m-y-k-purple-coral-lime".split("-")])
    for direction in range(np.max(directions)):
        idxs = np.argwhere(directions == direction + 1)
        plt.scatter(projected_data[0, idxs][:, 0], projected_data[1, idxs][:, 0], color = colors[direction], label = str(direction * 45))

    plt.legend()
    plt.title("Neuron Firing Data Plotted Using Two Principal Components")
    plt.savefig(str(figures_path))
    plt.show()
    

@app.command()
def main(
    data_path: Path = RAW_DATA_DIR / "ReachData.mat",
    figures_path: Path = FIGURES_DIR / "PCA Results.png"
):
    data = loadmat(str(data_path))
    directions = data['direction']
    neuron_firing = data['R']

    centered_neuron_firing = center_data(neuron_firing)
    cov = compute_covariance(centered_neuron_firing)
    evalues, evectors = compute_eigen(cov)

    print("Variance explained by first two principal components:", compute_two_principal_component_variance(evalues, evectors))
    visualize(centered_neuron_firing, directions, evectors, figures_path)



if __name__ == "__main__":
    app()
