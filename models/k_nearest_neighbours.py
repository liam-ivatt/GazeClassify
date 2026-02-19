import pandas as pd
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle

def hyperparameter_tuning(x_train, y_train):

    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=20)
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x_train, y_train)
    return grid_search.best_params_

def train_model(x_train, y_train, x_test, y_test, params):

    knn = KNeighborsClassifier(
        n_neighbors=params['n_neighbors'],
        weights=params['weights'],
        metric=params['metric'],
    )

    knn.fit(x_train, y_train)

    with open('prediction_models/knn.pkl', 'wb') as f:
        pickle.dump(knn, f)

    y_pred = knn.predict(x_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True
    )

    plt.title("Confusion Matrix - K-Nearest Neighbours")
    plt.show()

def main(x_train, y_train, x_test, y_test):

    params = hyperparameter_tuning(x_train, y_train)
    train_model(x_train, y_train, x_test, y_test, params)


