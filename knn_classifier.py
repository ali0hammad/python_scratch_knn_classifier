import math
import random
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_measure='euclidean', weights='uniform'):
        """
        Initializes the KNN Classifier.

        :param k: The number of nearest neighbors to consider.
        :param distance_measure: 'euclidean' or 'manhattan'.
        :param weights: 'uniform' (all neighbors have equal vote) or 'distance' (closer neighbors have more weight).
        """
        self.k = k
        self.distance_measure = distance_measure
        self.weights = weights
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.classes = []

    def load_and_split_data(self, X, y, test_ratio=0.3):
        """
        Splits the given dataset into training and testing sets.

        :param X: List of feature lists/tuples.
        :param y: List of labels.
        :param test_ratio: Float representing the proportion of data to use for testing (e.g., 0.3 for 30%).
        """
        # Combine X and y to shuffle them together
        dataset = list(zip(X, y))
        random.shuffle(dataset)

        # Calculate split index
        split_index = int(len(dataset) * (1 - test_ratio))

        # Split data
        train_data = dataset[:split_index]
        test_data = dataset[split_index:]

        # Unzip train data
        if train_data:
            self.X_train, self.y_train = zip(*train_data)
            self.X_train = list(self.X_train)
            self.y_train = list(self.y_train)

        # Unzip test data
        if test_data:
            self.X_test, self.y_test = zip(*test_data)
            self.X_test = list(self.X_test)
            self.y_test = list(self.y_test)

        # Store unique classes for the confusion matrix
        self.classes = list(set(y))
        self.classes.sort()

    def fit(self, X_train=None, y_train=None):
        """
        Fits the model with training data.
        If using load_and_split_data, calling this is not strictly necessary unless providing new train data.
        """
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.classes = list(set(y_train))
            self.classes.sort()

    def _calculate_distance(self, point1, point2):
        """
        Calculates the distance between two points based on the chosen measure.
        """
        if self.distance_measure == 'manhattan':
            return sum(abs(a - b) for a, b in zip(point1, point2))
        else: # Default to euclidean
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def predict(self, X_test=None):
        """
        Predicts the labels for the given test data based on K nearest neighbors.
        """
        if X_test is None:
            X_test = self.X_test

        predictions = []
        for test_point in X_test:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self._calculate_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))

            # Sort by distance
            distances.sort(key=lambda x: x[0])

            # Get top K neighbors
            neighbors = distances[:self.k]

            # Vote
            if self.weights == 'distance':
                votes = {}
                for dist, label in neighbors:
                    # Handle division by zero if distance is exactly 0
                    weight = 1.0 / dist if dist != 0 else float('inf')
                    votes[label] = votes.get(label, 0) + weight

                # Predict the class with the maximum vote weight
                predicted_class = max(votes, key=votes.get)
            else: # Default to uniform
                # Just count the frequency of labels
                labels = [label for dist, label in neighbors]
                most_common = Counter(labels).most_common(1)
                predicted_class = most_common[0][0]

            predictions.append(predicted_class)

        return predictions

    def evaluate(self, y_pred=None, y_true=None):
        """
        Compares predictions against ground truth and displays a confusion matrix.
        """
        if y_pred is None:
            y_pred = self.predict()
        if y_true is None:
            y_true = self.y_test

        # Initialize confusion matrix grid
        # matrix[actual][predicted]
        matrix = {actual: {predicted: 0 for predicted in self.classes} for actual in self.classes}

        # Populate the matrix
        for actual, predicted in zip(y_true, y_pred):
            matrix[actual][predicted] += 1

        # Print confusion matrix as a simple grid
        print("\n--- Confusion Matrix ---")

        # Header row
        header = f"{'Actual \\ Predicted':<20}" + "".join([f"{str(c):<12}" for c in self.classes])
        print(header)
        print("-" * len(header))

        # Data rows
        for actual in self.classes:
            row_str = f"{str(actual):<20}"
            for predicted in self.classes:
                row_str += f"{matrix[actual][predicted]:<12}"
            print(row_str)

        print("-" * len(header))

        # Calculate and print accuracy for convenience
        correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
        total = len(y_true)
        if total > 0:
            accuracy = correct / total
            print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
        else:
            print("No test data to evaluate.")
