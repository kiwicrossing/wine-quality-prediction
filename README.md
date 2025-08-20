# Wine Quality Software

This project uses deep learning on a large dataset of wines with varying qualities to determine the highest quality wines based on its chemical composition.

## Dataset Variables

The wine dataset contains 12 variables for each type of wine:
1. Fixed Acidity: The non-volatile acids in the wine (contributing to the tartness).
2. Volatile Acidity: The acetic acid content (contributing to the vinegar-like taste).
3. Citric Acid: One of the fixed acids in wine.
4. Residual Sugar: The sugar that remains after fermentation stops.
5. Chlorides: Contribute to the saltiness in wine.
6. Free Sulfur Dioxide: Added to the wine.
7. Total Sulfur Dioxide: The sum of bound and free sulfur dioxide.

## Project Structure

```bash
.
├── Makefile
├── README.md
├── example_screenshot.png
├── haarcascade_frontalface_default.xml
├── poetry.lock
├── pyproject.toml
├── recognition
│   ├── __init__.py
│   ├── facial_recognition.py
│   └── tests
│       ├── __init__.py
│       └── test_facial_recognition.py
├── src
│   ├── __init__.py
│   └── main.py
└── webcam_server
    ├── __init__.py
    └── capture_and_save.py
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

## Running the Webcam Capture Script (Windows Powershell)

This project was created to port images from the Windows-side webcam capture script, and send them
over to the WSL-side facial recognition software.

1. In Windows Powershell, navigate to the folder containing `capture_and_save.py`:
```powershell
cd "C:\path\to\webcam_server"
```
2. Run the script with the following command:
```powershell
python .\capture_and_save.py
```

This will continuously save webcam frames to a shared folder.

## Running the Facial Recognition Script (WSL)
1. Open your WSL terminal (Ubuntu-24.04).
2. Navigate to your project directory:
```bash
cd ~/facial-recognition
```
3. Run the main controller script:
```bash
python3 src/main.py
```

This script reads the latest frame from the shared folder and runs facial recognition using `recognition/facial_recognition.py`.

## Notes
- Ensure the shared folder (e.g. C:\Users\kiara\wsl-cam-share) is writable by your Windows user.
- WSL accesses Windows files via `/mnt/c/...`, so make sure paths in your WSL script reflect that.
- Since this project uses `cv2.imshow` in WSL, make sure that an X server is running and that you've set `export DISPLAY=:0`


## Author

Kiara Houghton, 2025

## Badges
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)