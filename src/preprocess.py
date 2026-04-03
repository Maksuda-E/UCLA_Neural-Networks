# Import pandas for DataFrame operations.
import pandas as pd

# Import train_test_split from sklearn for splitting the dataset.
from sklearn.model_selection import train_test_split

# Import configuration values used during preprocessing.
from src.config import TEST_SIZE, RANDOM_STATE, THRESHOLD

# Import the logger helper.
from src.logger import get_logger

# Import the custom exception class.
from src.custom_exception import ProjectException

# Create a logger for this module.
logger = get_logger(__name__)

# Define a function to clean and preprocess the raw dataset.
def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Start a try block to handle preprocessing errors.
    try:
        # Log the start of the cleaning process.
        logger.info("Starting data cleaning")

        # Create a copy of the input DataFrame so the original data is unchanged.
        df = data_frame.copy()

        # Remove extra spaces from column names.
        df.columns = df.columns.str.strip()

        # Remove duplicate rows from the dataset.
        df = df.drop_duplicates()

        # Convert the target column into binary classes using the threshold value.
        df["Admit_Chance"] = (df["Admit_Chance"] >= THRESHOLD).astype(int)

        # Drop the serial number column if it exists in the dataset.
        if "Serial_No" in df.columns:
            df = df.drop(columns=["Serial_No"])

        # Convert University_Rating into categorical type.
        df["University_Rating"] = df["University_Rating"].astype("object")

        # Convert Research into categorical type.
        df["Research"] = df["Research"].astype("object")

        # Apply one hot encoding to categorical columns.
        df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

        # Log the completion of data cleaning.
        logger.info("Data cleaning completed with shape %s", df.shape)

        # Return the cleaned DataFrame.
        return df

    # Catch any exception that happens during cleaning.
    except Exception as exc:
        # Log the preprocessing failure.
        logger.error("Data cleaning failed")

        # Raise a project-specific exception with details.
        raise ProjectException(f"Failed to clean data: {exc}")

# Define a function to separate features and target.
def split_features_target(data_frame: pd.DataFrame):
    # Start a try block for error handling.
    try:
        # Log the beginning of feature target splitting.
        logger.info("Splitting features and target")

        # Drop the target column from the feature set.
        x = data_frame.drop(columns=["Admit_Chance"])

        # Store the target column separately.
        y = data_frame["Admit_Chance"]

        # Return the feature matrix and target vector.
        return x, y

    # Catch any exception during feature target split.
    except Exception as exc:
        # Log the failure.
        logger.error("Feature target split failed")

        # Raise a custom exception with the original error.
        raise ProjectException(f"Failed to split features and target: {exc}")

# Define a function to split the data into train and test sets.
def split_train_test(x, y):
    # Start a try block for safe execution.
    try:
        # Log that train test split is starting.
        logger.info("Creating train test split")

        # Split the data while keeping class balance using stratify.
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Return the split datasets.
        return x_train, x_test, y_train, y_test

    # Catch any exception during train test split.
    except Exception as exc:
        # Log the failure.
        logger.error("Train test split failed")

        # Raise a custom exception with full details.
        raise ProjectException(f"Failed to split train and test data: {exc}")