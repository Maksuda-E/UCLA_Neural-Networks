# UCLA Admission Prediction using Neural Networks

## Overview
This project converts the Neural Networks notebook into a modular Python project and an interactive Streamlit web application.

The goal is to predict whether a student has a high or low chance of admission to UCLA using a Neural Network classification model.

The project follows the notebook workflow:

- Convert admission probability into classification using threshold (0.8)
- Perform data preprocessing and feature engineering
- Apply one-hot encoding for categorical variables
- Scale features using MinMaxScaler
- Train a Neural Network (MLPClassifier)
- Evaluate performance using classification metrics
- Deploy model using Streamlit web application

---

## Problem Statement
Universities receive a large number of applications every year.

Students want to estimate their chances of admission based on their profile.

Using machine learning, we predict whether a student has:

- High chance of admission  
- Low chance of admission  

based on academic and profile features.

---

## Dataset
The dataset contains the following features:

- GRE_Score: Score out of 340  
- TOEFL_Score: Score out of 120  
- University_Rating: Rating (1 to 5)  
- SOP: Statement of Purpose strength (1 to 5)  
- LOR: Letter of Recommendation strength (1 to 5)  
- CGPA: GPA out of 10  
- Research: 0 (No) or 1 (Yes)  
- Admit_Chance: Probability (0 to 1)  

Dataset file:  
`data/Admission.csv`

---

## Model Workflow
- Load dataset  
- Convert Admit_Chance into binary classification:
  - 1 if ≥ 0.80  
  - 0 if < 0.80  
- Drop unnecessary columns (Serial_No)  
- Convert categorical variables:
  - University_Rating  
  - Research  
- Apply one-hot encoding  
- Split data into train and test sets  
- Scale features using MinMaxScaler  
- Train Neural Network (MLPClassifier)  
- Evaluate model using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Confusion Matrix  
- Save model and artifacts  
- Use model in Streamlit app for prediction  

---

## Model Performance
- Training Accuracy: ~89.7%  
- Testing Accuracy: ~85%  

The model performs well but slightly below the 90% target due to dataset size and simple architecture.

---

## Prediction Interpretation
The model classifies students into:

- High Chance of Admission (1)  
- Low Chance of Admission (0)  

It also provides the probability of high admission.

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt


UCLA_Neural-Networks/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Admission.csv
│
├── artifacts/
│   ├── neural_network_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── metrics.json
│
├── logs/
│   └── project.log
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── custom_exception.py
    ├── logger.py
    ├── data_loader.py
    ├── preprocess.py
    ├── train.py
    └── predict.py