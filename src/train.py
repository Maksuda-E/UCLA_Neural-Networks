# This line imports os for folder creation
import os

# This line imports json for saving metrics
import json

# This line imports pickle for saving model objects
import pickle

# This line imports accuracy_score and confusion_matrix for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

# This line imports MLPClassifier for neural network classification
from sklearn.neural_network import MLPClassifier

# This line imports MinMaxScaler for feature scaling
from sklearn.preprocessing import MinMaxScaler

# This line imports configuration values
from src.config import (
    ARTIFACTS_DIR,
    MODEL_FILE_PATH,
    SCALER_FILE_PATH,
    FEATURE_COLUMNS_FILE_PATH,
    METRICS_FILE_PATH,
    RANDOM_STATE
)

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function scales the training and testing data
def scale_data(x_train, x_test):
    # This line starts the try block
    try:
        # This line logs scaling start
        logger.info("Feature scaling started")

        # This line creates the scaler
        scaler = MinMaxScaler()

        # This line fits the scaler on training data
        scaler.fit(x_train)

        # This line transforms the training data
        x_train_scaled = scaler.transform(x_train)

        # This line transforms the testing data
        x_test_scaled = scaler.transform(x_test)

        # This line logs scaling completion
        logger.info("Feature scaling completed successfully")

        # This line returns scaled data and scaler
        return x_train_scaled, x_test_scaled, scaler

    # This block handles scaling errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during feature scaling")

        # This line raises a custom exception
        raise ProjectException(f"Failed to scale data: {exc}")

# This function trains the neural network model
def train_model(x_train, y_train):
    # This line starts the try block
    try:
        # This line logs the start of model training
        logger.info("Neural network model training started")

        # This line creates the MLP classifier like the notebook
        model = MLPClassifier(
            hidden_layer_sizes=3,
            batch_size=50,
            max_iter=200,
            random_state=RANDOM_STATE
        )

        # This line trains the model on the training data
        model.fit(x_train, y_train)

        # This line logs that training completed successfully
        logger.info("Neural network model training completed successfully")

        # This line returns the trained model
        return model

    # This block handles training errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model training")

        # This line raises a custom exception
        raise ProjectException(f"Failed to train neural network model: {exc}")

# This function evaluates the trained model
def evaluate_model(model, x_train, y_train, x_test, y_test):
    # This line starts the try block
    try:
        # This line logs evaluation start
        logger.info("Model evaluation started")

        # This line makes predictions on training data
        ypred_train = model.predict(x_train)

        # This line makes predictions on testing data
        ypred_test = model.predict(x_test)

        # This line calculates training accuracy
        train_accuracy = accuracy_score(y_train, ypred_train)

        # This line calculates testing accuracy
        test_accuracy = accuracy_score(y_test, ypred_test)

        # This line calculates confusion matrix on test data
        cm = confusion_matrix(y_test, ypred_test)

        # This line creates a metrics dictionary
        metrics = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "confusion_matrix": cm.tolist()
        }

        # This line logs evaluation completion
        logger.info("Model evaluation completed successfully")

        # This line returns the metrics
        return metrics

    # This block handles evaluation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model evaluation")

        # This line raises a custom exception
        raise ProjectException(f"Failed to evaluate model: {exc}")

# This function saves the model artifacts
def save_artifacts(model, scaler, feature_columns, metrics):
    # This line starts the try block
    try:
        # This line creates the artifacts folder if it does not exist
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # This line opens the model file in write binary mode
        with open(MODEL_FILE_PATH, "wb") as model_file:
            # This line saves the model
            pickle.dump(model, model_file)

        # This line opens the scaler file in write binary mode
        with open(SCALER_FILE_PATH, "wb") as scaler_file:
            # This line saves the scaler
            pickle.dump(scaler, scaler_file)

        # This line opens the feature columns file in write binary mode
        with open(FEATURE_COLUMNS_FILE_PATH, "wb") as feature_file:
            # This line saves the feature columns
            pickle.dump(feature_columns, feature_file)

        # This line opens the metrics file in write mode
        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as metrics_file:
            # This line saves the metrics
            json.dump(metrics, metrics_file, indent=4)

        # This line logs successful artifact saving
        logger.info("Artifacts saved successfully")

    # This block handles saving errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while saving artifacts")

        # This line raises a custom exception
        raise ProjectException(f"Failed to save artifacts: {exc}")