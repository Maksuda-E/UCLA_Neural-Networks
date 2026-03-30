# This line imports pickle for loading saved objects
import pickle

# This line imports pandas for creating input DataFrame
import pandas as pd

# This line imports saved artifact paths
from src.config import MODEL_FILE_PATH, SCALER_FILE_PATH, FEATURE_COLUMNS_FILE_PATH

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function loads the saved model, scaler, and feature columns
def load_artifacts():
    # This line starts the try block
    try:
        # This line opens the model file
        with open(MODEL_FILE_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        # This line opens the scaler file
        with open(SCALER_FILE_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # This line opens the feature columns file
        with open(FEATURE_COLUMNS_FILE_PATH, "rb") as feature_file:
            feature_columns = pickle.load(feature_file)

        # This line logs successful loading
        logger.info("Model, scaler, and feature columns loaded successfully")

        # This line returns the loaded artifacts
        return model, scaler, feature_columns

    # This block handles loading errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while loading artifacts")

        # This line raises a custom exception
        raise ProjectException(f"Failed to load model artifacts: {exc}")

# This function converts user input into the format needed by the model
def prepare_input_data(user_input: dict, feature_columns: list):
    # This line starts the try block
    try:
        # This line creates a DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # This line applies one hot encoding to the categorical fields
        input_df = pd.get_dummies(input_df, columns=["University_Rating", "Research"], dtype=int)

        # This line matches input columns to training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # This line returns the prepared input
        return input_df

    # This block handles input preparation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while preparing input data")

        # This line raises a custom exception
        raise ProjectException(f"Failed to prepare input data: {exc}")

# This function predicts admission status
def predict_admission(user_input: dict):
    # This line starts the try block
    try:
        # This line loads the artifacts
        model, scaler, feature_columns = load_artifacts()

        # This line prepares the input data
        input_df = prepare_input_data(user_input, feature_columns)

        # This line scales the input data
        input_scaled = scaler.transform(input_df)

        # This line predicts the result
        prediction = model.predict(input_scaled)[0]

        # This line returns a human readable result
        if prediction == 1:
            return "High Chance of Admission"

        # This line returns the negative result
        return "Low Chance of Admission"

    # This block handles prediction errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during prediction")

        # This line raises a custom exception
        raise ProjectException(f"Failed to predict admission status: {exc}")