# BA305-Project
Group Project for BA305


Questions:

1) In our dataset what are the most important factors that influence Fraud vs Non-Fraud?


2) Predict whether a transaction is fraudulent or not based on the given features using different methods? Determine which method is best for this dataset (try 2 methods) and compare the results.
   - 2a) For the best method, perform hyperparameter tuning to optimize the model's performance.

General Details:

- Dataset: Credit Card Fraud Detection Dataset (https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)

- This is a classification problem where the target variable is whether a transaction is fraudulent (1) or not (0).

- The dataset is highly imbalanced and very large, with a very small percentage of fraudulent transactions compared to non-fraudulent ones.

- Current methods are PCA, Logistic Regression, and Random Forest (decision trees)
    - PCA is used for dimensionality reduction and feature extraction
    - Logistic Regression and Random Forest are used for classification.

- Evaluation metrics to consider: Accuracy, Precision, Recall, F1 Score, and AUC-ROC.
    - Prioritize Precision and Recall for the fraudulent class due to the imbalanced nature of the dataset.
    - We mainly care about identifying fraudulent transactions correctly (high Recall) while minimizing false positives (high Precision).

- Potential other methods are SMOTE + Gradient Boosting (XGBoost/LightGBM) - Claude recommended this as a potential method for outlier detection and handling imbalanced datasets. 
