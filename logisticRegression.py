from sklearn.datasets import load_iris
#https://www.geeksforgeeks.org/iris-dataset/

# Load dataset and preprocess
iris = load_iris()
X = iris.data[:, 0:2]  # Select only two features for 2D visualization, [:, 0:2] means all rows and first two columns
print(X.shape)      #shape is given by (rows, columns)
y = iris.target

print(y.shape) #150 samples/rows. 1 column that represents the type of flower the features represent. The class it belongs to 

feature_names = iris.feature_names[:2]
print(feature_names)