# Getting started

## 1. Cloning the repository
To clone this repository on your local machine, please run:
```
git clone [url]
cd decoding-bci
```

## 2. Download all requirements
To download the necessary libraries and dependencies, please run:
```
pip install requirements.txt
```

## 3. Process data
All neuron firing data has already been processed and saved in `data/processed`. If you would like to process the data yourself, please run:
```
python -m decoding_bci.dataset
```

## 4. PCA Analysis
To run PCA on the processed data, please run:
```
python -m decoding_bci.pca_analysis.pca_manual
```
This code will output a graph showing the trials as graphed by the first two principal components, with each color representing the direction of the motor movement in degrees. It will also output the variance explained by the first two principla components. The graph is saved in `reports/figures`

## 5. Neural Network Training and Prediction
To train a neural network on the processed data, please run:
```
python -m decoding_bci.modeling.train
```
The code will display an embedding of the neural firing data after training, which will be saved in `reports/figures`. Model weights are saved in `models` under the name `model.keras`.
To predict on the test dataset using the trained model, please run:
```
python -m decoding_bci.modeling.predict
```
Predictions are saved under `data/processed/test_predictions.csv`