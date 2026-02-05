import pandas as pd
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pickle

def hyperparameter_tuning(x_train, y_train):

    param_grid = {
        'max_depth': [3, 5, 7, 10, 15],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['gini', 'entropy'],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=20)
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)

def train_model(x_train, y_train, x_test, y_test):

    dt = DecisionTreeClassifier(
        random_state=20,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='gini', )

    dt.fit(x_train, y_train)

    with open('../prediction_models/dtree.pkl', 'wb') as f:
        pickle.dump(dt, f)

    y_pred = dt.predict(x_test)

    plot_tree(dt)
    plt.show()

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True
    )

    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

def main():

    data = pd.read_csv("../dataset.csv", index_col=False)
    x = data.drop(["label"], axis=1).to_numpy()
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=20
    )

    # hyperparameter_tuning(x_train, y_train)
    train_model(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()

