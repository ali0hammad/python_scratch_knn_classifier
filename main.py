import random
from knn_classifier import KNNClassifier

def generate_mock_data(num_samples=100):
    """
    Generates some simple mock 2D data for classification.
    Class A is centered around (2, 2)
    Class B is centered around (8, 8)
    """
    X = []
    y = []
    for _ in range(num_samples // 2):
        # Class A points
        x1 = random.uniform(0, 4)
        x2 = random.uniform(0, 4)
        X.append([x1, x2])
        y.append('A')

        # Class B points
        x1 = random.uniform(6, 10)
        x2 = random.uniform(6, 10)
        X.append([x1, x2])
        y.append('B')

    return X, y

def main():
    print("Generating mock data...")
    X, y = generate_mock_data(100)

    print("\n--- Testing KNN Classifier ---")

    # 1. Initialize classifier
    k_value = 3
    test_ratio = 0.3 # 30% test data

    print(f"Hyperparameters:")
    print(f"K = {k_value}")
    print(f"Distance Measure = 'euclidean'")
    print(f"Weights = 'uniform'")
    print(f"Test split = {int(test_ratio * 100)}-{int((1 - test_ratio) * 100)}")

    knn = KNNClassifier(k=k_value, distance_measure='euclidean', weights='uniform')

    # 2. Split data
    knn.load_and_split_data(X, y, test_ratio=test_ratio)
    print(f"\nData split: {len(knn.X_train)} train points, {len(knn.X_test)} test points.")

    # 3. Predict and evaluate
    print("\nPredicting and evaluating...")
    knn.evaluate()

    print("\n--- Testing with different hyperparameters ---")

    k_value_2 = 5
    print(f"Hyperparameters:")
    print(f"K = {k_value_2}")
    print(f"Distance Measure = 'manhattan'")
    print(f"Weights = 'distance'")

    knn2 = KNNClassifier(k=k_value_2, distance_measure='manhattan', weights='distance')
    knn2.load_and_split_data(X, y, test_ratio=0.4) # 40% test
    knn2.evaluate()

if __name__ == "__main__":
    main()
