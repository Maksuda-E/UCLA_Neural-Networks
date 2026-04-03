# Import the dataset path from the configuration file.
from src.config import DATA_FILE_PATH

# Import the function that loads the dataset.
from src.data_loader import load_data

# Import preprocessing functions for cleaning and splitting the data.
from src.preprocess import clean_data, split_features_target, split_train_test

# Import training functions for scaling, training, evaluation, and saving artifacts.
from src.train import scale_data, train_model, evaluate_model, save_artifacts


# Define the full training pipeline function.
def run_training_pipeline() -> dict:
    # Load the dataset from the configured file path.
    data_frame = load_data(DATA_FILE_PATH)

    # Clean and preprocess the raw dataset.
    cleaned_data = clean_data(data_frame)

    # Separate the features and target variable.
    features, target = split_features_target(cleaned_data)

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = split_train_test(features, target)

    # Scale the training and testing feature data.
    x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

    # Train the neural network model using the scaled training data.
    model = train_model(x_train_scaled, y_train)

    # Evaluate the trained model using both train and test data.
    metrics = evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test)

    # Save the trained model, scaler, feature names, and metrics.
    save_artifacts(model, scaler, list(features.columns), metrics)

    # Return the evaluation results.
    return metrics


# Check whether this file is being run directly.
if __name__ == "__main__":
    # Run the full training pipeline and store the results.
    results = run_training_pipeline()

    # Print a success message after the training pipeline completes.
    print("Training pipeline completed successfully.")

    # Print the evaluation results to the console.
    print(results)