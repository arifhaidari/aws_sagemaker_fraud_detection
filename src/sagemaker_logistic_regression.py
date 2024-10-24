import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

if __name__ == "__main__":
     # Set up argument parsing
     parser = argparse.ArgumentParser()
     
     # SageMaker specific arguments for channels and model saving
     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
     parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

     # Parse the arguments
     # args = parser.parse_args() # this does not work
     args, _ = parser.parse_known_args()

     # Debugging step: print file paths
     print(f"Training data path: {os.path.join(args.train, 'train.csv')}")
     
     try:
          # Load the training data
          train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
          print(f"Training data loaded successfully with shape: {train_data.shape}")
     except Exception as e:
          print(f"Error loading training data: {e}")
          raise
     
     # Ensure that the target variable exists
     if 'is_fraud' not in train_data.columns:
          raise ValueError("Target variable 'is_fraud' not found in training data")

     # Split into features and labels
     X_train = train_data.drop('is_fraud', axis=1)
     y_train = train_data['is_fraud']

     # Check for non-numeric data (optional depending on your dataset)
     # Example: handle categorical features with one-hot encoding (if necessary)
     X_train = pd.get_dummies(X_train)

     print("Training model...")
     try:
          # Train the model
          model = LogisticRegression()
          model.fit(X_train, y_train)
     except Exception as e:
          print(f"Error during model training: {e}")
          raise
     
     # Save the trained model
     print("Saving model...")
     try:
          joblib.dump(model, os.path.join(args.model_dir, "logistic_regression_model.joblib"))
          print("Model saved successfully!")
     except Exception as e:
          print(f"Error saving model: {e}")
          raise
