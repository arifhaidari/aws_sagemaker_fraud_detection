import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     
     # SageMaker specific arguments
     parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
     # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
     parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
     # parser.add_argument('--train-file', type=str, default='train.csv')
     # parser.add_argument('--test-file', type=str, default='test.csv')
     
     # we can also pass our custom argurments such as hyperparamters and so on
     # parser.add_argument('--n_estimators', type=int, default=100)
     # parser.add_argument('--random_state', type=int, default=0)
     
     args = parser.parse_args()
     print('got here now 1')
     print(f"Train data path: {os.path.join(args.train, 'train.csv')}")

     # Load training data
     train_data = pd.read_csv(os.path.join(args.train, "train.csv"))
     # we can also write in the following way
     # train_data = pd.read_csv(os.path.join(args.train, args.train_file))

     X_train = train_data.drop('is_fraud')
     y_train = train_data['is_fraud']

     # Train the model
     print("Training model...")
     model = LogisticRegression()
     model.fit(X_train, y_train)

     # Save the model to the output directory for deployment
     print("Saving model...")
     joblib.dump(model, os.path.join(args.model_dir, "logistic_regression_model.joblib"))
     print("Model saved successfully!")
