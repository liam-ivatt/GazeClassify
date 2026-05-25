import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

def hyperparameter_tuning(x_train, y_train):

    param_grid = {
        'hidden_layer_sizes': [
            (10,), (20,), (30,),
            (20, 10), (50, 25), (100, 50),
            (50, 25, 10), (100, 50, 25)
        ],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.001, 0.01],
        'solver': ['adam', 'sgd'],
    }

    mlp = MLPClassifier(
        early_stopping=True,
        n_iter_no_change=10,
        tol=0.001,
        max_iter=500,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=20)
    grid_search = GridSearchCV(mlp, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)

    return grid_search.best_params_

def train_model(x_train, y_train, x_test, y_test, params):

    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        learning_rate_init=params['learning_rate_init'],
        alpha=params['alpha'],
        solver=params['solver'],
        early_stopping=True,
        n_iter_no_change=10,
        tol=0.001,
        max_iter=500,
    )
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True,
        display_labels=["centre", "left", "right"]
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame.from_dict(report)
    df.to_csv('evaluation/mlp_report_new.csv')

    plt.title("Confusion Matrix - MLP")
    plt.show()

    return mlp

def one_participant_out():

    df = pd.read_csv("data/dataset_test_logo.csv")
    features = ["right_relative_x", "right_relative_y", "left_relative_x", "left_relative_y"]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    per_participant = []
    all_y_test = []
    all_y_pred = []

    X = df[features]
    y = df["label"]

    print("Training MLP")

    for participant in df["participant"].unique():
        test = df[df["participant"] == participant]
        train = df[df["participant"] != participant]

        X_train = train[features].values
        y_train = train["label"].values
        X_test = test[features].values
        y_test = test["label"].values

        params = hyperparameter_tuning(X_train, y_train)

        mlp = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            learning_rate_init=params['learning_rate_init'],
            alpha=params['alpha'],
            solver=params['solver'],
            early_stopping=True,
            n_iter_no_change=10,
            tol=0.001,
            max_iter=500,
        )

        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        df_report = pd.DataFrame.from_dict(report).transpose()
        df_report['participant'] = participant
        per_participant.append(df_report)


    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        learning_rate_init=params['learning_rate_init'],
        alpha=params['alpha'],
        solver=params['solver'],
        early_stopping=True,
        n_iter_no_change=10,
        tol=0.001,
        max_iter=500,
    )

    mlp.fit(X, y)
    joblib.dump(mlp, "prediction_models/mlp.mdl")

    agg_report = classification_report(all_y_test, all_y_pred, target_names=le.classes_, output_dict=True)
    df_agg = pd.DataFrame.from_dict(agg_report).transpose()
    df_agg['participant'] = 'aggregate'

    df_all = pd.concat([*per_participant, df_agg])
    df_all.index.name = 'class'
    df_all = df_all.reset_index().set_index(['participant', 'class'])
    df_all.to_csv('evaluation/mlp_logo_report.csv')

    ConfusionMatrixDisplay.from_predictions(
        all_y_test, all_y_pred,
        display_labels=le.classes_,
        cmap='Blues', colorbar=True
    )
    plt.title("Confusion Matrix - Multi-Layer Perceptron")
    plt.show()

def main():
    one_participant_out()