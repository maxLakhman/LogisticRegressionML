from sklearn.datasets import load_iris


# Load dataset and preprocess
iris = load_iris()
X = iris.data[:, 0:2]  # Select only two features for 2D visualization, must select better features X = iris.data[:, :2]
print(X.shape)