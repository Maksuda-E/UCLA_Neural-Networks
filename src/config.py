# Import the os module to work with file and folder paths.
import os

# Get the absolute path of the current file.
CURRENT_FILE_PATH = os.path.abspath(__file__)

# Get the src folder path from the current file path.
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)

# Get the main project directory from the src folder path.
PROJECT_DIR = os.path.dirname(SRC_DIR)

# Define the data folder path.
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Define the artifacts folder path where model files will be stored.
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# Define the logs folder path where log files will be stored.
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# Define the dataset file path.
DATA_FILE_PATH = os.path.join(DATA_DIR, "Admission.csv")

# Define the trained model file path.
MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, "neural_network_model.pkl")

# Define the scaler file path.
SCALER_FILE_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# Define the feature columns file path.
FEATURE_COLUMNS_FILE_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

# Define the metrics file path.
METRICS_FILE_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# Define the application log file path.
LOG_FILE_PATH = os.path.join(LOGS_DIR, "project.log")

# Set the random state so results are reproducible.
RANDOM_STATE = 123

# Set the test size for train test split.
TEST_SIZE = 0.20

# Set the target threshold for converting admission chance into binary classes.
THRESHOLD = 0.80