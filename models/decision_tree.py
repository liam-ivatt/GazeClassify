import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, GridSearchCV

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

    return grid_search.best_params_

def train_model(x_train, y_train, x_test, y_test, params):

    dt = DecisionTreeClassifier(
        criterion=params['criterion'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf = params['min_samples_leaf'])

    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    plot_tree(dt)
    plt.show()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap='Blues',
        colorbar=True
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame.from_dict(report)
    df.to_csv('evaluation/dt_report.csv')

    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    return dt

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

    print("Training DT")

    for participant in df["participant"].unique():
        test = df[df["participant"] == participant]
        train = df[df["participant"] != participant]

        X_train = train[features].values
        y_train = train["label"].values
        X_test = test[features].values
        y_test = test["label"].values

        params = hyperparameter_tuning(X_train, y_train)

        dt = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            criterion=params['criterion'],
        )

        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        df_report = pd.DataFrame.from_dict(report).transpose()
        df_report['participant'] = participant
        per_participant.append(df_report)


    dt = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        criterion=params['criterion'],
    )

    dt.fit(X, y)
    joblib.dump(dt, "prediction_models/dt.mdl")

    agg_report = classification_report(all_y_test, all_y_pred, target_names=le.classes_, output_dict=True)
    df_agg = pd.DataFrame.from_dict(agg_report).transpose()
    df_agg['participant'] = 'aggregate'

    df_all = pd.concat([*per_participant, df_agg])
    df_all.index.name = 'class'
    df_all = df_all.reset_index().set_index(['participant', 'class'])
    df_all.to_csv('evaluation/dt_logo_report.csv')

    ConfusionMatrixDisplay.from_predictions(
        all_y_test, all_y_pred,
        display_labels=le.classes_,
        cmap='Blues', colorbar=True
    )
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

def main():

    one_participant_out()
    # train_model(x_train, y_train, x_test, y_test, params)