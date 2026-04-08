

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("Primeiras linhas do dataset: ")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scarler = StandardScaler()
X_train_scaled = scarler.fit_transform(X_train)
X_test_scaled = scarler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("\nAcurácia: ", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação: ")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nMatriz de Confusão: ")
print(confusion_matrix(y_test, y_pred))
