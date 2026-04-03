# Import os for file and folder operations.
import os

# Import json for saving metrics in JSON format.
import json

# Import pickle for saving Python objects such as model and scaler.
import pickle

# Import evaluation metrics from sklearn.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import the neural network classifier.
from sklearn.neural_network import MLPClassifier

# Import the MinMaxScaler used in the notebook pipeline.
from sklearn.preprocessing import MinMaxScaler

# Import configuration values and artifact paths.
from src.config import (
    ARTIFACTS_DIR,
    MODEL_FILE_PATH,
    SCALER_FILE_PATH,
    FEATURE_COLUMNS_FILE_PATH,
    METRICS_FILE_PATH,
    RANDOM_STATE
)

# Import the logger helper.
from src.logger import get_logger

# Import the custom exception class.
from src.custom_exception import ProjectException

# Create a logger for this module.
logger = get_logger(__name__)

# Define a function to scale training and testing data.
def scale_data(x_train, x_test):
    # Start a try block to catch scaling errors.
    try:
        # Log that scaling has started.
        logger.info("Starting feature scaling")

        # Create a MinMaxScaler object.
        scaler = MinMaxScaler()

        # Fit the scaler on the training data only.
        scaler.fit(x_train)

        # Transform the training data using the fitted scaler.
        x_train_scaled = scaler.transform(x_train)

        # Transform the testing data using the same scaler.
        x_test_scaled = scaler.transform(x_test)

        # Log that scaling has completed.
        logger.info("Feature scaling completed")

        # Return the scaled train data, scaled test data, and scaler object.
        return x_train_scaled, x_test_scaled, scaler

    # Catch any exception during scaling.
    except Exception as exc:
        # Log the scaling failure.
        logger.error("Feature scaling failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to scale data: {exc}")

# Define a function to train the neural network model.
def train_model(x_train, y_train):
    # Start a try block to catch training errors.
    try:
        # Log that model training has started.
        logger.info("Starting model training")

        # Create the MLPClassifier using the notebook configuration.
        model = MLPClassifier(
             hidden_layer_sizes=(10,),   # Increase neurons
             batch_size=50,
             max_iter=500,               # Increase iterations
             random_state=RANDOM_STATE,
             early_stopping=True         # Stops automatically when no improvement
       )

        # Fit the model on the training data.
        model.fit(x_train, y_train)

        # Log that model training has completed.
        logger.info("Model training completed")

        # Return the trained model.
        return model

    # Catch any exception during training.
    except Exception as exc:
        # Log the training failure.
        logger.error("Model training failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to train neural network model: {exc}")

# Define a function to evaluate the trained model.
def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Start a try block to catch evaluation errors.
    try:
        # Log that evaluation has started.
        logger.info("Starting model evaluation")

        # Predict the labels for the training data.
        y_train_pred = model.predict(x_train)

        # Predict the labels for the testing data.
        y_test_pred = model.predict(x_test)

        # Create a dictionary of evaluation metrics.
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "precision": float(precision_score(y_test, y_test_pred)),
            "recall": float(recall_score(y_test, y_test_pred)),
            "f1_score": float(f1_score(y_test, y_test_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }

        # Log the evaluation metrics.
        logger.info("Evaluation completed with metrics %s", metrics)

        # Return the metrics dictionary.
        return metrics

    # Catch any exception during evaluation.
    except Exception as exc:
        # Log the evaluation failure.
        logger.error("Model evaluation failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to evaluate model: {exc}")

# Define a function to save the trained model, scaler, columns, and metrics.
def save_artifacts(model, scaler, feature_columns, metrics):
    # Start a try block to handle file save errors.
    try:
        # Log that artifact saving has started.
        logger.info("Saving artifacts")

        # Create the artifacts directory if it does not already exist.
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # Open the model file path in binary write mode.
        with open(MODEL_FILE_PATH, "wb") as model_file:
            # Save the trained model object.
            pickle.dump(model, model_file)

        # Open the scaler file path in binary write mode.
        with open(SCALER_FILE_PATH, "wb") as scaler_file:
            # Save the scaler object.
            pickle.dump(scaler, scaler_file)

        # Open the feature columns file path in binary write mode.
        with open(FEATURE_COLUMNS_FILE_PATH, "wb") as feature_file:
            # Save the feature column names.
            pickle.dump(feature_columns, feature_file)

        # Open the metrics file path in text write mode.
        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as metrics_file:
            # Save the metrics dictionary as formatted JSON.
            json.dump(metrics, metrics_file, indent=4)

        # Log that artifacts were saved successfully.
        logger.info("Artifacts saved successfully")

    # Catch any exception during saving.
    except Exception as exc:
        # Log the save failure.
        logger.error("Artifact saving failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to save artifacts: {exc}")