import numpy as np

class SimpleSVM:
    def __init__(self, learning_rate=0.001, regularization_strength=0.01, num_epochs=1000):
        # Initialize SVM hyperparameters
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_epochs = num_epochs
        self.weights = None  # This will store the weights for each feature
        self.bias = 0  # Initialize bias term as zero

    def fit(self, data, labels):
        # Determine the number of features based on the shape of the input data
        num_features = data.shape[1]
        # Initialize weights to zero, with one weight for each feature
        self.weights = np.zeros(num_features)
        
        # Perform gradient descent for the specified number of epochs
        for epoch in range(self.num_epochs):
            # Loop over each data point
            for i, x in enumerate(data):
                # Calculate the condition to check if the current point is correctly classified
                condition = labels[i] * (np.dot(x, self.weights) + self.bias)
                
                # If the condition is met (i.e., point is correctly classified)
                if condition >= 1:
                    # Only apply regularization to weights (no change to bias)
                    self.weights -= self.learning_rate * (2 * self.regularization_strength * self.weights)
                else:
                    # Apply penalty (update weights and bias) for misclassification
                    # Adjust weights: regularization and moving in direction of label * input
                    self.weights -= self.learning_rate * (2 * self.regularization_strength * self.weights - np.dot(x, labels[i]))
                    # Update bias by moving it in the direction of the label
                    self.bias -= self.learning_rate * labels[i]

    def predict(self, x):
        # Predict the class of a single data point
        # np.dot calculates the linear combination of weights and features
        # np.sign gives +1 if positive and -1 if negative
        return np.sign(np.dot(x, self.weights) + self.bias)

    def predict_batch(self, data):
        # Predict the class for a batch of data points
        # np.dot(data, self.weights) + self.bias applies the linear decision function to each point
        # np.sign converts each score to +1 or -1 based on the sign of the result
        return np.sign(np.dot(data, self.weights) + self.bias)

    def get_parameters(self):
        # Return the learned weights and bias for external inspection or interpretation
        return self.weights, self.bias

# Step 1: Define the data and labels
data = np.array([
    [2, 3],  # Point A, Class +1
    [3, 3],  # Point B, Class +1
    [6, 5],  # Point C, Class -1
    [7, 8]   # Point D, Class -1
])
labels = np.array([1, 1, -1, -1])

# Step 2: Create and train the SVM model
svm = SimpleSVM(learning_rate=0.001, regularization_strength=0.01, num_epochs=1000)
svm.fit(data, labels)

# Step 3: Test the SVM with new points
test_points = np.array([
    [4, 4],  # New point to classify
    [3, 3],  # Point from the training data
    [6, 6]   # New point
])
predictions = svm.predict_batch(test_points)

# Display predictions for each test point
for point, prediction in zip(test_points, predictions):
    print(f"Point {point} is classified as {'Class +1' if prediction == 1 else 'Class -1'}")

# Display learned parameters
weights, bias = svm.get_parameters()
print("Learned weights:", weights)
print("Learned bias:", bias)
