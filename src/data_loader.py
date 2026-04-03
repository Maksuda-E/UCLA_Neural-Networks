# Import pandas for reading the CSV dataset.
import pandas as pd

# Import the custom logger helper.
from src.logger import get_logger

# Import the custom project exception class.
from src.custom_exception import ProjectException

# Create a logger for this module.
logger = get_logger(__name__)

# Define a function to load the dataset from a CSV file.
def load_data(file_path: str) -> pd.DataFrame:
    # Start a try block to catch loading errors.
    try:
        # Write a log message before loading the dataset.
        logger.info("Loading dataset from %s", file_path)

        # Read the dataset from the given file path.
        data_frame = pd.read_csv(file_path)

        # Write a log message after successful loading.
        logger.info("Dataset loaded successfully with shape %s", data_frame.shape)

        # Return the loaded DataFrame.
        return data_frame

    # Catch any exception that happens during file loading.
    except Exception as exc:
        # Write an error message to the log file.
        logger.error("Failed to load dataset")

        # Raise a project-specific exception with the original error message.
        raise ProjectException(f"Failed to load data: {exc}")