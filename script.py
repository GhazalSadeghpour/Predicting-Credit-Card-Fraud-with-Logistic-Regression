import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions[transactions["isFraud"] == 1].shape[0])

# Summary statistics on amount column
print(transactions[["amount"]].describe())


# Create isPayment field
transactions['isPayment'] = np.where((transactions['type'] ==  'DEBIT') | (transactions['type'] == 'PAYMENT'), 1, 0)

# Create isMovement field
transactions['isMovement'] = np.where((
  (transactions['type'] ==  'CASH_OUT') | (transactions['type'] ==  'TRANSFER')
),1,0)

# Create accountDiff field
transactions['accountDiff'] = transactions['oldbalanceDest'] - transactions['oldbalanceOrg']

# Create features and label variables
features = ['amount', 'isPayment', 'isMovement', 'accountDiff']

label = 'isFraud'

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(transactions[features], transactions[label], test_size=0.3)


# Normalize the features variables
ssc = StandardScaler()
X_train_scaled = ssc.fit_transform(X_train) 
X_test_scaled = ssc.transform(X_test)  

# Fit the model to the training data
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

# Score the model on the training data
training_score = lr.score(X_train_scaled, y_train)
print(f"Training score: {training_score}")


# Score the model on the test data
test_score = lr.score(X_test_scaled, y_test)
print(f"Test score:  {test_score}")

# Print the model coefficients
print(f"Model coefficients: {lr.coef_}")

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction


# Combine new transactions into a single array


# Normalize the new transactions


# Predict fraud on the new transactions


# Show probabilities on the new transactions