# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and separate it into feature matrix and target labels.

2. Split the data into training and testing sets.

3. Standardize the feature values using a scaler to improve model performance.

4. Train an SGD Classifier on the training data and evaluate it using predictions on the test set.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data                # independent variables
y = iris.target              # dependent variable (0,1,2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = SGDClassifier(loss="log_loss", max_iter=1000, learning_rate="optimal")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

Developed by: Jesron Shawn C J
RegisterNumber:  25012933
*/
```

## Output:
<img width="932" height="311" alt="image" src="https://github.com/user-attachments/assets/a8e3bd7b-9c4a-447b-a9e7-0edb6d00f428" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
