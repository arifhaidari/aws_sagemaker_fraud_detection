# Deploying Machine Learning Model (Fraud Detection) - AWS SageMaker

## Project Overview

The purpose of this project is to develop a machine learning model to detect fraudulent credit card transactions and deploy it on AWS SageMaker for production use. The goal is to build a robust model to identify fraud transactions and prevent potential financial loss.

---

### Dataset

The dataset used for this project contains simulated credit card transactions, with both legitimate and fraud transactions from the duration of **1st Jan 2019 - 31st Dec 2020**. The dataset includes 1000 customers making transactions with 800 merchants.

- [Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Columns:**

- `trans_date_trans_time`: Date and time of the transaction
- `cc_num`: Credit card number (masked)
- `merchant`: Merchant name
- `category`: Transaction category
- `amt`: Transaction amount
- ... _(other features like customer details, geographical data, etc.)_

The key feature for classification is:

- `is_fraud`: 1 (fraud transaction), 0 (legitimate transaction)

---

### Problem Definition

The objective is to predict whether a transaction is fraudulent (`is_fraud=1`) or legitimate (`is_fraud=0`). This is a **binary classification problem**, and the project involves various steps of data preprocessing, feature engineering, and model building.

---

### Project Structure

1. **Data Preprocessing**:

   - Handled missing values and cleaned irrelevant data.
   - Balanced the dataset using **undersampling** of the non-fraudulent class:
     ```python
     fraud = df[df["is_fraud"] == 1]
     not_fraud = df[df["is_fraud"] == 0]
     not_fraud = not_fraud.sample(fraud.shape[0])
     df = pd.concat([fraud, not_fraud])
     ```

2. **Exploratory Data Analysis (EDA)**:

   - Calculated the percentage of fraud vs. non-fraud transactions:
     ```python
     fraud_ratio = df['is_fraud'].mean() * 100
     print(f"Fraud transactions percentage: {fraud_ratio:.2f}%")
     ```

3. **Modeling**:

   - Several algorithms were considered for binary classification, including **Logistic Regression**, **SVM**, and **XGBoost**:
     - **Logistic Regression** was selected initially for its simplicity and efficiency in binary classification tasks.
     - **XGBoost** was explored later for improved performance on non-linear relationships and handling class imbalance.

4. **Balancing the Dataset**:
   Given the significant class imbalance:

   - **Fraud transactions**: 9651
   - **Non-fraud transactions**: 1842743

   The **undersampling** method was used to balance the majority class (`non-fraud`) and the minority class (`fraud`). This helps the model focus better on identifying frauds but at the cost of some data loss:

   ```python
   not_fraud = not_fraud.sample(fraud.shape[0])
   df = pd.concat([fraud, not_fraud])
   ```

5. **Model Saving**:
   After training the model (Random Forest or Logistic Regression), it was saved to a folder for future use:
   ```python
   import os
   import joblib
   os.makedirs('../models', exist_ok=True)
   joblib.dump(model, '../models/fraud_detection_model.pkl')
   ```

---

### Model Evaluation and Deployment

The trained model will be evaluated using metrics like **accuracy**, **precision**, **recall**, and **F1-score** to ensure proper detection of fraudulent transactions. Once optimized, it will be deployed on **AWS SageMaker** to enable real-time predictions in a cloud environment.

---

### Techniques Used

- **Under-sampling**: To balance the dataset and prevent the model from being biased towards the non-fraudulent transactions.
- **Logistic Regression**: A simple and interpretable model for binary classification.
- **XGBoost**: For handling complex patterns and class imbalance effectively.
- **AWS SageMaker**: For scalable and secure deployment of the model.

---

### Future Work

- Further optimization of the model using **SMOTE** or other techniques to generate synthetic samples for the minority class (fraud).
- Deployment and monitoring of the model on **AWS SageMaker** for real-time transaction fraud detection.

---

### Credits

This project was developed using the simulated credit card transaction dataset and inspired by **Brandon Harris**'s Sparkov Data Generation tool on GitHub.
