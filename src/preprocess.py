# This line imports pandas for data processing
import pandas as pd

# This line imports train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# This line imports configuration values
from src.config import TEST_SIZE, RANDOM_STATE

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function cleans the dataset
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # This line starts the try block
    try:
        # This line logs the start of data cleaning
        logger.info("Starting data cleaning")

        # This line creates a copy of the dataset
        df = df.copy()

        # This line removes leading and trailing spaces from column names
        df.columns = df.columns.str.strip()

        # This line removes duplicate rows
        df = df.drop_duplicates()

        # This line converts the target variable into binary form like the notebook
        df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

        # This line drops the serial number column if it exists
        if "Serial_No" in df.columns:
            df = df.drop(["Serial_No"], axis=1)

        # This line converts University_Rating to object type if it exists
        if "University_Rating" in df.columns:
            df["University_Rating"] = df["University_Rating"].astype("object")

        # This line converts Research to object type if it exists
        if "Research" in df.columns:
            df["Research"] = df["Research"].astype("object")

        # This line creates dummy variables for categorical columns like the notebook
        df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

        # This line logs that data cleaning is complete
        logger.info("Data cleaning completed successfully")

        # This line returns the cleaned dataset
        return df

    # This block handles errors during cleaning
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during data cleaning")

        # This line raises a custom exception
        raise ProjectException(f"Failed to clean data: {exc}")

# This function separates features and target
def split_features_target(df: pd.DataFrame):
    # This line starts the try block
    try:
        # This line logs the start of feature and target split
        logger.info("Splitting features and target")

        # This line stores all columns except Admit_Chance in x
        x = df.drop(["Admit_Chance"], axis=1)

        # This line stores the Admit_Chance column in y
        y = df["Admit_Chance"]

        # This line logs successful split
        logger.info("Feature and target split completed successfully")

        # This line returns x and y
        return x, y

    # This block handles split errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while splitting features and target")

        # This line raises a custom exception
        raise ProjectException(f"Failed to split features and target: {exc}")

# This function splits the data into training and testing sets
def split_train_test(x, y):
    # This line starts the try block
    try:
        # This line logs the start of train test split
        logger.info("Starting train test split")

        # This line splits x and y into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # This line logs that splitting is complete
        logger.info("Train test split completed successfully")

        # This line returns the split data
        return x_train, x_test, y_train, y_test

    # This block handles splitting errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during train test split")

        # This line raises a custom exception
        raise ProjectException(f"Failed to split train and test data: {exc}")