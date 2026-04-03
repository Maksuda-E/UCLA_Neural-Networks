# UCLA Neural Networks Admission Predictor

This project predicts whether a student has a high or low chance of admission to UCLA using a neural network classification model.

## Project Objective

The original notebook uses `Admit_Chance` as the target variable.

For classification, the target is converted into binary classes:

- 1 means high chance of admission when `Admit_Chance >= 0.80`
- 0 means low chance of admission when `Admit_Chance < 0.80`

## Project Structure

- `app.py` contains the Streamlit web application
- `main.py` runs the training pipeline
- `src/data_loader.py` loads the dataset
- `src/preprocess.py` cleans and prepares the data
- `src/train.py` trains and evaluates the model
- `src/predict.py` loads saved artifacts and predicts new inputs

## How to Run the Project

Install dependencies:

```bash
pip install -r requirements.txt