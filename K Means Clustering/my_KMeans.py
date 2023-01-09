import pandas as pd
import numpy as np

# Used the algorithms from the slides
class my_KMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol
        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def euclidean_distance(self, p, q):
        # get the number of data attributes
        attributes = self.num_columns
        sum = 0.0
        for attribute in range(attributes):
            # take the sum of the absolute value of pi-qi to the power of 2
            # where i equals values from 1 to the number of attributes(n)
            sum += ((p[attribute] - q[attribute]) ** 2)
        # then raise the sum to the power of (1/2)
        distance = sum ** (1 / 2)
        return distance

    def k_means_plus_plus(self, X):
        self.num_rows = X.shape[0]
        self.num_columns = X.shape[1]
        # (1) Select an initial point at random to be the first centroid
        cluster_centers = [X[np.random.randint(self.num_rows)]]
        # (2) For k – 1 steps
        for steps in range(self.n_clusters - 1):
            # (3) for each of the N points, need to loop through centroids,
            # find the minimum squared distance to the currently selected centroids
            min_squared_distance = np.array([np.min([self.euclidean_distance(row, centroids)
                                                     for centroids in cluster_centers]) for row in X])
            # (4) Randomly select a new centroid by choosing a point with probability proportional
            probs = min_squared_distance / sum(min_squared_distance)
            all_probs = probs.cumsum()
            random = np.random.rand()
            for index, prob in enumerate(all_probs):
                if random < prob:
                    wanted_index = index
                    break
            cluster_centers.append(X[wanted_index])
        return cluster_centers

    def random(self, X):
        self.num_rows = X.shape[0]
        self.num_columns = X.shape[1]
        np.random.RandomState(self.max_iter)
        random_idx = np.random.permutation(X.shape[0])
        cluster_centers = X[random_idx[:self.n_clusters]]
        return cluster_centers

    def centroid_guide(self, X):
        if self.init == "k-means++":
            cluster_centers = self.k_means_plus_plus(X)
        elif self.init == "random":
            cluster_centers = self.random(X)
        else:
            raise Exception("Unknown value of self.init.")
        return cluster_centers

    def fit_helper(self, X):
        last_inertia = None
        # (1) select initial centroids based on init
        cluster_centers = self.centroid_guide(X)
        # (2) repeat until centroids don't change (by more than tol)
        for loop in range(self.max_iter + 1):
            inertia = 0
            clusters = [[] for i in range(self.n_clusters)]
            # (3) for k clusters by assigning all points to the closest centroid
            for row in X:
                distances = [self.euclidean_distance(row, center) for center in cluster_centers]
                # calculate the inertia
                inertia += (min(distances) ** 2)
                # index of the cluster row and add the row to the cluster
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(row)
            if (last_inertia and last_inertia - inertia < self.tol) or loop == self.max_iter:
                break
            # (4) recompute the centroid of each cluster
            old_cluster_centers =  np.copy(cluster_centers)
            cluster_centers = self.recompute_cluster(clusters)
            last_inertia = inertia
        return cluster_centers, inertia

    def recompute_cluster(self, clusters):
        cluster_centers = np.zeros((self.n_clusters, self.num_columns))
        for cluster_index, cluster_data in enumerate(clusters):
            cluster_data_mean = np.mean(clusters[cluster_index], axis=0)
            cluster_centers[cluster_index] = cluster_data_mean
        return cluster_centers

    def fit(self, X):
        # X: pd.DataFrame, independent variables, float
        # repeat self.n_init times and keep the best run
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        self.X_feature = X.to_numpy()
        for i in range(self.n_init):
            cluster_centers, inertia = self.fit_helper(self.X_feature)
            if self.inertia_ == None or inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = cluster_centers
        return

    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        dists = [[self.euclidean_distance(x, centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def fit_predict(self, X):
        self.X = X
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions



