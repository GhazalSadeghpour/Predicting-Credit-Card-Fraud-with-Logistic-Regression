# Predicting Credit Card Fraud with Logistic Regression

## Overview
This project, part of Codecademy's **Machine Learning/AI Engineer** path, predicts fraudulent credit card transactions using **Logistic Regression**.

## Dataset & Features
The dataset (`transactions_modified.csv`) includes transaction details like amount, type, and account balances. Key engineered features:
- **isPayment**: 1 for DEBIT/PAYMENT, else 0
- **isMovement**: 1 for CASH_OUT/TRANSFER, else 0
- **accountDiff**: Difference between destination and origin balances

## Model Training
- **Data Split**: 70% training, 30% test
- **Scaling**: StandardScaler normalizes features
- **Classifier**: Logistic Regression
- **Evaluation**: Model scores printed for training and test sets

## Fraud Prediction
After training, the model predicts fraud in new transactions:
```python
lr.predict(sample_transactions)
lr.predict_proba(sample_transactions)
```

## Running the Code
Run the script in Python:
```bash
python fraud_detection.py
```

## Future Improvements
- Try other ML models (e.g., Random Forest, Neural Networks)
- Tune hyperparameters
- Address class imbalance

## Author
Codecademy Machine Learning/AI Engineer Path

