# Wine Quality Software
![wine-glass.svg](wine-glass.svg)

This project uses **deep learning** on a large dataset of wines with varying qualities to
determine the highest quality wines based on its chemical composition.

## Project Structure

```bash
.
├── Makefile
├── README.md
├── codecov.yml
├── coverage.xml
├── dataset
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
├── model
│   ├── __init__.py
│   ├── data_loading.py
│   ├── neural_network.py
│   ├── sql_analytics.py
│   └── tests
│       ├── __init__.py
│       ├── test_data_loading.py
│       └── test_neural_network.py
├── output
│   ├── accuracy_per_epoch.png
│   ├── alcohol_distribution.png
│   ├── database_info
│   │   ├── feature_correlations.csv
│   │   └── wine_data.db
│   ├── feature_correlations.png
│   ├── loss_per_epoch.png
│   └── training_history_table.png
├── poetry.lock
├── pyproject.toml
└── src
    ├── __init__.py
    └── main.py
```

## Installation

Use [poetry](https://python-poetry.org/) for dependency management.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

It is not required, but recommended to use a virtual environment to run this application.
To create a virtual environment, run `poetry env activate` inside the top-level directory.
To enter the virtual environment, copy and paste the last command's output.
If you would like to exit the virtual environment, run `deactivate`.

## How to Run

To run this project, simply use `make` or `make run`.
It will generate a histogram of the alcohol distribution between red and white wines,
which determines how common each alcohol level is among the wines.

## Generated Graphs

After running the project, several graphs and visualizations are generated and saved in the `output/` directory:

- **Alcohol Distribution Histogram:**  
  A side-by-side histogram comparing the alcohol content distribution between red and white wines.

- **Training History Table:**  
  A table (as a PNG image) showing the training and validation loss and accuracy for each epoch.

- **Loss per Epoch:**  
  A line plot showing how the training and validation loss change over each epoch.

- **Accuracy per Epoch:**  
  A line plot showing how the training and validation accuracy change over each epoch.

## Dataset Information

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

The dataset that this project uses is from the [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/).

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. [Online]. Available: https://doi.org/10.24432/C56S3T.

## Example Use Cases

- **Wine Type Classification:**
  Predict whether a wine is red or white based on its chemical properties using the trained neural network.

- **Quality Prediction:**
  Extend the model to predict wine quality scores, helping winemakers or distributors assess product quality automatically.

- **Batch Quality Control:**
  Use the model to flag batches of wine that are likely to be of low quality before bottling, saving time and resources.

- **Data Visualization:**
  Use the generated graphs to present insights about the dataset, such as how alcohol content varies between red and white wines.

- **Model Performance Monitoring:**
  Track training and validation accuracy/loss over time to detect overfitting or underfitting and tune model hyperparameters accordingly.

## Author

Kiara Houghton, 2025

## Badges
[![codecov](https://codecov.io/github/kiwicrossing/wine-quality-prediction/graph/badge.svg?token=G2RBXAYXXH)](https://codecov.io/github/kiwicrossing/wine-quality-prediction)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Poetry](https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D)


![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)


![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)