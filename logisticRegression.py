from sklearn.datasets import load_iris
#https://www.geeksforgeeks.org/iris-dataset/
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)

# Load dataset and preprocess
iris = load_iris()
X = iris.data[:, 0:2]  # Select only two features for 2D visualization, [:, 0:2] means all rows and first two columns, features are sepal length and width
print(X.shape)      #shape is given by (rows, columns)

y = (iris.target != 0).astype(int)  # Binary classification (Class 0 vs Not Class 0)

feature_names = iris.feature_names[:2]
print(feature_names)
print("Translation, im dealing w 150 samples of sepal length, width accordingly")

#y = iris.target
print(y.shape) #150 samples/rows. 1 column that contains the type of flower the features represent. The class it belongs to 

#Now, split the data into more neat datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the statistical model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)

#Train the model on the training data
model.fit(X_train, y_train)

#Predict the labels of the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))