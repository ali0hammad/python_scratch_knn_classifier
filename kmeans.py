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


class KMeans:
    def __init__(self, data, split=70, k=3, metric="euclidean", max_iter=100):
        self.k = k
        self.metric = metric
        self.max_iter = max_iter

        indices = list(range(len(data)))
        random.shuffle(indices)

        n_train = int(len(indices) * split / 100)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        self.X_train = []
        for i in train_idx:
            self.X_train.append(data[i])

        self.X_test = []
        for i in test_idx:
            self.X_test.append(data[i])

        self.centroids = random.sample(self.X_train, k)

    def fit(self):
        for _ in range(self.max_iter):
            clusters = []
            for _ in range(self.k):
                clusters.append([])

            for point in self.X_train:
                best_index = 0
                best_dist = dist(point, self.centroids[0], self.metric)
                for i in range(1, self.k):
                    d = dist(point, self.centroids[i], self.metric)
                    if d < best_dist:
                        best_dist = d
                        best_index = i
                clusters[best_index].append(point)

            new_centroids = []
            for i, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    new_centroids.append(self.centroids[i])
                    continue

                dims = len(cluster[0])
                center = []
                for d in range(dims):
                    total = 0
                    for point in cluster:
                        total += point[d]
                    center.append(total / len(cluster))
                new_centroids.append(center)

            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids

        return self.centroids

    def predict(self, point):
        best_index = 0
        best_dist = dist(point, self.centroids[0], self.metric)
        for i in range(1, self.k):
            d = dist(point, self.centroids[i], self.metric)
            if d < best_dist:
                best_dist = d
                best_index = i
        return best_index

    def silhouette(self):
        labels = []
        for point in self.X_test:
            labels.append(self.predict(point))

        scores = []
        for i, point in enumerate(self.X_test):
            same_cluster = []
            for j in range(len(self.X_test)):
                if labels[j] == labels[i] and j != i:
                    same_cluster.append(self.X_test[j])

            if len(same_cluster) > 0:
                total = 0
                for other in same_cluster:
                    total += dist(point, other, self.metric)
                a = total / len(same_cluster)
            else:
                a = 0

            b = None
            for cluster_id in set(labels):
                if cluster_id == labels[i]:
                    continue

                other_cluster = []
                for j in range(len(self.X_test)):
                    if labels[j] == cluster_id:
                        other_cluster.append(self.X_test[j])

                if len(other_cluster) == 0:
                    continue

                total = 0
                for other in other_cluster:
                    total += dist(point, other, self.metric)
                avg_dist = total / len(other_cluster)

                if b is None or avg_dist < b:
                    b = avg_dist

            if b is None:
                b = 0

            denom = max(a, b)
            if denom == 0:
                s = 0
            else:
                s = (b - a) / denom

            scores.append(s)

        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0

        print(f"Silhouette Score: {avg_score:.4f}")
        return avg_score


if __name__ == "__main__":
    data = [[1, 2], [2, 1], [2, 3], [8, 8], [9, 8], [8, 9],
            [5, 5], [6, 5], [5, 6], [1, 1], [9, 9], [5, 4]]

    km = KMeans(data, split=70, k=3, metric="euclidean", max_iter=100)
    km.fit()
    km.silhouette()
