import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# 2. Split features (X) and target (y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train model
model = SVC(gamma='auto')
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")   

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample data
train_x = np.array([[1], [2], [3], [4], [5]])
train_y = np.array([2, 4, 5, 4, 5])

# Train model
regr = LinearRegression()
regr.fit(train_x, train_y)

# Predict
test_x = np.array([[6]])
predicted = regr.predict(test_x)

# Evaluate
print(f"R2 Score: {r2_score([5], regr.predict(np.array([[5]])))}")   