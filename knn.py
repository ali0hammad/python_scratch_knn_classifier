import random
import math


def dist(a, b, m="euclidean"):
    if m == "manhattan":
        total = 0
        for x, y in zip(a, b):
            total += abs(x - y)
        return total
    else:
        total = 0
        for x, y in zip(a, b):
            total += (x - y) ** 2
        return math.sqrt(total)


class KNN:
    def __init__(self, data, labels, split=70, k=3, metric="euclidean", weighted=False):
        self.k = k
        self.metric = metric
        self.weighted = weighted

        indices = list(range(len(data)))
        random.shuffle(indices)

        n_train = int(len(indices) * split / 100)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        self.X_train = []
        self.y_train = []
        for i in train_idx:
            self.X_train.append(data[i])
            self.y_train.append(labels[i])

        self.X_test = []
        self.y_test = []
        for i in test_idx:
            self.X_test.append(data[i])
            self.y_test.append(labels[i])

    def predict(self, point):
        distances = []
        for x, y in zip(self.X_train, self.y_train):
            d = dist(point, x, self.metric)
            distances.append((d, y))

        distances.sort(key=lambda pair: pair[0])
        nearest = distances[:self.k]

        votes = {}
        for d, label in nearest:
            if self.weighted:
                weight = 1 / (d + 1e-9)
            else:
                weight = 1

            if label not in votes:
                votes[label] = 0
            votes[label] += weight

        best_label = None
        best_score = -1
        for label, score in votes.items():
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def evaluate(self):
        predictions = []
        for x in self.X_test:
            predictions.append(self.predict(x))

        all_labels = sorted(set(self.y_train) | set(self.y_test))

        confusion = {}
        for actual in all_labels:
            confusion[actual] = {}
            for predicted in all_labels:
                confusion[actual][predicted] = 0

        for predicted, actual in zip(predictions, self.y_test):
            confusion[actual][predicted] += 1

        correct = 0
        for label in all_labels:
            correct += confusion[label][label]

        if len(self.y_test) > 0:
            accuracy = correct / len(self.y_test)
        else:
            accuracy = 0

        print("Confusion Matrix (rows=actual, cols=predicted):")
        header = "\t".join(str(l) for l in all_labels)
        print("\t" + header)

        for actual in all_labels:
            row = [str(confusion[actual][predicted]) for predicted in all_labels]
            print(str(actual) + "\t" + "\t".join(row))

        print(f"Accuracy: {accuracy:.2f}")
        return confusion, accuracy


if __name__ == "__main__":
    data = [[1, 2], [2, 1], [2, 3], [8, 8], [9, 8], [8, 9], [5, 5], [6, 5], [5, 6]]
    labels = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]

    knn = KNN(data, labels, split=70, k=3, metric="euclidean", weighted=True)
    knn.evaluate()
