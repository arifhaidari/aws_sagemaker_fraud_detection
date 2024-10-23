# Deploying Machine Learning Model (Fraud Detection) - AWS SageMaker (Under Development)

the purpose of this project is to develop the model and then deploy in the AWS Cloud Environment

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

About the Dataset
This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

Source of Simulation
This was generated using Sparkov Data Generation | Github tool created by Brandon Harris. This simulation was run for the duration - 1 Jan 2019 to 31 Dec 2020. The files were combined and converted into a standard format.

Problem Definition
Objective: The goal of this project is to detect fraudulent transactions using machine learning techniques. This is a binary classification problem where the model predicts whether a transaction is fraud (is_fraud=1) or legitimate (is_fraud=0).
Target Variable: is_fraud (1 = fraud, 0 = legitimate or non-fraud)

Algorithms used for modeling:
Key Points for Choosing Between SVM, Logistic Regression and XGBoost:
SVM:
details and comparison goes here

Logistic Regression:
Simple, interpretable, and fast to train.
Performs well if the data is linearly separable.
However, may not perform as well on complex patterns without feature engineering.

XGBoost:
More complex but can capture non-linear relationships in the data.
Handles missing values and class imbalance more effectively.
Might be slower to train but typically yields higher performance, especially on large datasets.

---

Over-sampling the minority class (fraud): You can duplicate fraud samples to balance the dataset.
Synthetic data generation (SMOTE): Synthetic Minority Over-sampling Technique (SMOTE) creates synthetic samples of the minority class to balance the dataset.
Using model techniques: Some models, like Random Forest or XGBoost, have built-in handling for imbalanced data using class weighting.
undersampling:
In fraud detection, there is typically a large imbalance between fraud (minority class) and non-fraud (majority class) transactions. In your case, the outputs show:

Fraud transactions: 2145
Non-fraud transactions: 553574
This is highly imbalanced. Machine learning models, particularly classification models like Logistic Regression, tend to perform poorly on imbalanced datasets because they are biased toward the majority class (non-fraud) and may fail to accurately detect fraud.
code for under-sampling:
not_fraud = not_fraud.sample(fraud.shape[0])
df = pd.concat([fraud, not_fraud])

The purpose of this code is to balance the class distribution by under-sampling the majority (non-fraud) class. While it may improve model performance on imbalanced data, it can lead to loss of information from the majority class, which might impact the model's ability to generalize well. You should carefully consider the trade-offs or explore alternative methods.
