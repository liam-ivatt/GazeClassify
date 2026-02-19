from sklearn.neural_network import MLPClassifier
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle

def hyperparameter_tuning(x_train, y_train):

    param_grid = {
        'hidden_layer_sizes': [5, 10, 15, 20, 25],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.001, 0.01],
        'solver': ['adam', 'sgd'],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=20)
    grid_search = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x_train, y_train)
    return grid_search.best_params_

def train_model(x_train, y_train, x_test, y_test, params):

    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        learning_rate_init=params['learning_rate_init'],
        alpha=params['alpha'],
        solver=params['solver'],
    )
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True
    )

    plt.title("Confusion Matrix - MLP")
    plt.show()

def main(x_train, y_train, x_test, y_test):

    params = hyperparameter_tuning(x_train, y_train)
    train_model(x_train, y_train, x_test, y_test, params)