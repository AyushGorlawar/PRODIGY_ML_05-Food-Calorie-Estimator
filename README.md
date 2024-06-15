# PRODIGY_ML_05

# Food Calorie Estimator

This project aims to develop a model that can accurately recognize food items from images and estimate their calorie content.

## Directory Structure
food_calorie_estimator/
├── data/
│ └── food-101/
├── models/
│ └── model.pth
├── notebooks/
│ └── Food_Calorie_Estimation.ipynb
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
├── requirements.txt
└── README.md

## Setup

1. Clone the repository.
2. Download the Food-101 dataset and extract it to the `data/` directory.
3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Train the model:
    ```sh
    python src/train.py
    ```
5. Evaluate the model:
    ```sh
    python src/evaluate.py
    ```

## Usage

- Use the Jupyter notebook in `notebooks/` for exploratory data analysis and model prototyping.
- Train the model using `train.py` and evaluate it using `evaluate.py`.
Setting Up VS Code
Open VS Code.
Open the root directory of your project (food_calorie_estimator).
Create the files and directories as outlined above.
Install the Python extension for VS Code if you haven't already.
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
