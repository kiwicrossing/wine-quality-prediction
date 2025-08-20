# Wine Quality Software

This project uses deep learning on a large dataset of wines with varying qualities to
determine the highest quality wines based on its chemical composition.

## Dataset Variables

The wine dataset contains 12 variables for each type of wine:
1. Fixed Acidity: The non-volatile acids in the wine (contributing to the tartness).
2. Volatile Acidity: The acetic acid content (contributing to the vinegar-like taste).
3. Citric Acid: One of the fixed acids in wine.
4. Residual Sugar: The sugar that remains after fermentation stops.
5. Chlorides: Contribute to the saltiness in wine.
6. Free Sulfur Dioxide: Added to the wine.
7. Total Sulfur Dioxide: The sum of bound and free sulfur dioxide.
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality

## Project Structure

```bash
.
├── Makefile
├── README.md
├── model
│   ├── __init__.py
│   ├── data_loading.py
│   └── neural_network.py
├── output
│   └── alcohol_distribution.png
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