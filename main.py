import joblib
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

    # Supervised Models
    # decision_tree()
    # random_forest()
    # k_nearest_neighbours()
    # mlp()

    # Unsupervised Models
    # k_means_clustering(data)
    # hierarchical_clustering(data)

    model = joblib.load("prediction_models/dt.mdl")

    print(dir(model))

if __name__ == "__main__":
    main()


