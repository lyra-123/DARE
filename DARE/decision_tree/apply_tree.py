import os
import pickle
import numpy as np


class DecisionTreeApplier:

    def __init__(self, tree_path):

        if not os.path.exists(tree_path):
            raise FileNotFoundError(
                f"Decision tree model not found: {tree_path}"
            )

        with open(tree_path, "rb") as f:
            self.tree = pickle.load(f)


    def predict(self, feature):

        feature = np.asarray(feature)

        if feature.ndim == 1:
            feature = feature.reshape(1, -1)

        model_id = self.tree.predict(feature)[0]

        return int(model_id)



def load_tree(tree_path):

    return DecisionTreeApplier(tree_path)



if __name__ == "__main__":

    TREE_PATH = "./save_tree.pkl"

    example_feature = np.random.randn(8)

    tree = DecisionTreeApplier(
        TREE_PATH
    )

    model_id = tree.predict(
        example_feature
    )

    print(model_id)