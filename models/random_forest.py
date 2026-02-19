import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle

def hyperparameter_tuning(x_train, y_train):

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=20)
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x_train, y_train)

    return grid_search.best_params_

def train_model(x_train, y_train, x_test, y_test, params):

    rf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        criterion=params['criterion'],)

    rf.fit(x_train, y_train)

    with open('prediction_models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)

    y_pred = rf.predict(x_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True
    )

    plt.title("Confusion Matrix - Random Forest")
    plt.show()

def main(x_train, y_train, x_test, y_test):

    params = hyperparameter_tuning(x_train, y_train)
    train_model(x_train, y_train, x_test, y_test, params)

