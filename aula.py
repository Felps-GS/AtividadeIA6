# ===============================================
# Aula 06: Prática em Python 
# Tema: Machine Learning - Classificação
# Dataset: Iris
# ===============================================


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_tree = tree.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)

print("\nAcurácia Árvore de Decisão: ", accuracy_score(y_test, y_pred_tree))
print("\nAcurácia KNN: ", accuracy_score(y_test, y_pred_knn))

cm_tree = confusion_matrix(y_test, y_pred_tree)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=iris.target_names)
disp_tree.plot()
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.show()


cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=iris.target_names)
disp_knn.plot()
plt.title("Matriz de Confusão - KNN")
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, title):
    X = X[:, 0]
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('#FFAAAA', '#AAFFAA', '#AAAAFF')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(('#FF0000', '#00FF00', '#0000FF')))
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.show()

X_vis = X.iloc[:, [:, :2]]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42, stratify=y)

scarler_vis = StandardScaler()
X_train_scaled_vis = scarler_vis.fit_transform(X_train_vis)
X_test_scaled_vis = scarler_vis.transform(X_test_vis)

tree_vis = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_vis.fit(X_train_scaled_vis, y_train_vis)

knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_train_scaled_vis, y_train_vis)

plot_decision_boundary(X_train_scaled_vis, y_train_vis, tree_vis, "Fronteira de Decisão - Árvore")
plot_decision_boundary(X_train_scaled_vis, y_train_vis, knn_vis, "Fronteira de Decisão - KNN")




    

