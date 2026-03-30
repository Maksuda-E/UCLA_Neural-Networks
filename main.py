# This line imports the dataset file path
from src.config import DATA_FILE_PATH

# This line imports the data loading function
from src.data_loader import load_data

# This line imports preprocessing functions
from src.preprocess import clean_data, split_features_target, split_train_test

# This line imports model functions
from src.train import scale_data, train_model, evaluate_model, save_artifacts

# This line imports the logger
from src.logger import get_logger

# This line creates a logger for this file
logger = get_logger(__name__)

# This function controls the full training pipeline
def main():
    # This line starts the try block
    try:
        # This line logs pipeline start
        logger.info("Main pipeline started")

        # This line loads the dataset
        df = load_data(DATA_FILE_PATH)

        # This line cleans the dataset
        df_clean = clean_data(df)

        # This line splits features and target
        x, y = split_features_target(df_clean)

        # This line splits the data into train and test sets
        x_train, x_test, y_train, y_test = split_train_test(x, y)

        # This line scales the data
        x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

        # This line trains the model
        model = train_model(x_train_scaled, y_train)

        # This line evaluates the model
        metrics = evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test)

        # This line saves the artifacts
        save_artifacts(model, scaler, list(x.columns), metrics)

        # This line prints success message
        print("Training completed successfully")

        # This line prints the metrics heading
        print("Model evaluation results")

        # This line loops through metrics
        for key, value in metrics.items():
            # This line prints each metric
            print(f"{key}: {value}")

        # This line logs pipeline completion
        logger.info("Main pipeline completed successfully")

    # This block handles pipeline errors
    except Exception as exc:
        # This line logs pipeline failure
        logger.error(f"Main pipeline failed: {exc}")

        # This line raises the error again
        raise

# This line checks if the file is run directly
if __name__ == "__main__":
    # This line calls the main function
    main()