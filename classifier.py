import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold

data = pd.read_csv("dataset.csv")
x = data.drop(["label"], axis=1).to_numpy()
y = data["label"]

x_train, x_test, y_train, y_test = ms.train_test_split(
    x, y, test_size=0.2, random_state=20
)

dt = tree.DecisionTreeClassifier(
    random_state=20,
    max_depth=3,)

kf = KFold(n_splits=5, shuffle=True, random_state=20)
scores = ms.cross_val_score(dt, x_train, y_train, cv=5)
print(scores)

dt.fit(x_train, y_train)

with open('prediction_models/dtree.pkl', 'wb') as f:
    pickle.dump(dt, f)

print(x_test.shape)

y_pred = dt.predict(x_test)

tree.plot_tree(dt)
plt.show()

disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap='Blues',
    colorbar=True
)

plt.title("Confusion Matrix - Decision Tree")
plt.show()
