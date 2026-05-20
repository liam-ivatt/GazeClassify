import pandas as pd
from sklearn.model_selection import train_test_split

# IMPORT PREPROCESSING MODULES
from dataset import main as build_dataset

# IMPORT MODELS
from models.decision_tree import main as decision_tree
from models.random_forest import main as random_forest
from models.k_nearest_neighbours import main as k_nearest_neighbours
from models.mlp import main as mlp
from models.k_means_clustering import main as k_means_clustering
from models.hierarchical_clustering import main as hierarchical_clustering

def main():

    data_path = "data/logo"

    # build_dataset(data_path)

    data = pd.read_csv(f"data/dataset_test_logo.csv", index_col=False)
    x = data.drop(["label", "participant"], axis=1).to_numpy()
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=20
    )

    # Supervised Models
    decision_tree(x_train, y_train, x_test, y_test)
    random_forest(x_train, y_train, x_test, y_test)
    k_nearest_neighbours(x_train, y_train, x_test, y_test)
    mlp(x_train, y_train, x_test, y_test)

    # Unsupervised Models
    # k_means_clustering(data)
    # hierarchical_clustering(data)

if __name__ == "__main__":
    main()


