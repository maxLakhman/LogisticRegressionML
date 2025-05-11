from sklearn.datasets import load_iris
#https://www.geeksforgeeks.org/iris-dataset/

# Load dataset and preprocess
iris = load_iris()

X = iris.data[:, 0:2]  # Select only two features for 2D visualization, [:, 0:2] means all rows and first two columns
print(X.shape)      #shape is given by (rows, columns)
feature_names = iris.feature_names[:2]
print(feature_names)
print("Translation, im dealing w 150 samples of sepal length, width accordingly")

y = iris.target
print(y.shape) #150 samples/rows. 1 column that represents the type of flower the features represent. The class it belongs to 

#Now, split the data into more neat datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)