# Import pickle for loading saved Python objects.
import pickle

# Import pandas for creating the input DataFrame.
import pandas as pd

# Import artifact file paths from the configuration module.
from src.config import MODEL_FILE_PATH, SCALER_FILE_PATH, FEATURE_COLUMNS_FILE_PATH

# Import the logger helper.
from src.logger import get_logger

# Import the custom exception class.
from src.custom_exception import ProjectException

# Create a logger for this module.
logger = get_logger(__name__)

# Define a function to load the trained model, scaler, and feature columns.
def load_artifacts():
    # Start a try block to catch artifact loading errors.
    try:
        # Open the saved model file in binary read mode.
        with open(MODEL_FILE_PATH, "rb") as model_file:
            # Load the trained model object.
            model = pickle.load(model_file)

        # Open the saved scaler file in binary read mode.
        with open(SCALER_FILE_PATH, "rb") as scaler_file:
            # Load the scaler object.
            scaler = pickle.load(scaler_file)

        # Open the saved feature columns file in binary read mode.
        with open(FEATURE_COLUMNS_FILE_PATH, "rb") as feature_file:
            # Load the feature column list.
            feature_columns = pickle.load(feature_file)

        # Log that artifacts were loaded successfully.
        logger.info("Artifacts loaded successfully")

        # Return the loaded model, scaler, and feature columns.
        return model, scaler, feature_columns

    # Catch any exception during artifact loading.
    except Exception as exc:
        # Log the loading failure.
        logger.error("Artifact loading failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to load model artifacts: {exc}")

# Define a function to transform raw user input into model-ready data.
def prepare_input_data(user_input: dict, feature_columns: list):
    # Start a try block to catch input preparation errors.
    try:
        # Convert the user input dictionary into a one row DataFrame.
        input_df = pd.DataFrame([user_input])

        # Convert University_Rating into categorical type.
        input_df["University_Rating"] = input_df["University_Rating"].astype("object")

        # Convert Research into categorical type.
        input_df["Research"] = input_df["Research"].astype("object")

        # Apply one hot encoding to the categorical input columns.
        input_df = pd.get_dummies(input_df, columns=["University_Rating", "Research"], dtype=int)

        # Reindex the encoded input so it matches the training feature columns exactly.
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Return the prepared input DataFrame.
        return input_df

    # Catch any exception during input preparation.
    except Exception as exc:
        # Log the preparation failure.
        logger.error("Input preparation failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to prepare input data: {exc}")

# Define a function that predicts admission from user input.
def predict_admission(user_input: dict):
    # Start a try block to catch prediction errors.
    try:
        # Load the saved model artifacts.
        model, scaler, feature_columns = load_artifacts()

        # Prepare the raw user input into model-ready format.
        input_df = prepare_input_data(user_input, feature_columns)

        # Scale the prepared input using the saved scaler.
        input_scaled = scaler.transform(input_df)

        # Predict the class label from the scaled input.
        predicted_class = int(model.predict(input_scaled)[0])

        # Check whether the model supports probability prediction.
        if hasattr(model, "predict_proba"):
            # Predict the probability of the positive class.
            probability_of_high_admission = float(model.predict_proba(input_scaled)[0][1])
        else:
            # Fall back to class value if probability is unavailable.
            probability_of_high_admission = float(predicted_class)

        # Convert the numeric class into a readable label.
        predicted_label = "High Chance of Admission" if predicted_class == 1 else "Low Chance of Admission"

        # Return all prediction outputs in dictionary form.
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "probability_of_high_admission": probability_of_high_admission
        }

    # Catch any exception during prediction.
    except Exception as exc:
        # Log the prediction failure.
        logger.error("Prediction failed")

        # Raise a custom exception with details.
        raise ProjectException(f"Failed to predict admission status: {exc}")