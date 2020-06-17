import pandas as pd

df = pd.read_csv('data/classified_data.csv')
df.head()

# Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_predictors = df.drop('TARGET CLASS', axis=1)
scaler.fit(df_predictors)

scaled_predictors = scaler.transform(df_predictors)

# Train model
X = pd.DataFrame(scaled_predictors, columns=df.columns[:-1])
y = df['TARGET CLASS']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101
)

# Evaluating K Value
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

error_rate = []

def predict_y(k = 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return  knn.predict(X_test)

for i in range(1,100):
    y_pred = predict_y(i)
    error_rate.append(np.mean(y_pred != y_test))

# Plot results
import matplotlib.pyplot as plt

def plot_results():
    plt.figure(figsize=(20,6))
    plt.plot(
        range(1,100),
        error_rate,
        color='blue',
        linestyle='dashed',
        marker='o',
        markerfacecolor='red',
        markersize=5
    )

plot_results()

minimum_error = np.min(error_rate)
optimal_k = error_rate.index(minimum_error) + 1
y_pred = predict_y(optimal_k)

# Evaluate predictions
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))