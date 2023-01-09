import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", P=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.P = P

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classifications = np.unique(y)
        # stores the data to be used later on
        self.stored_X = X
        self.stored_y = y
        return

    def minkowski_distance(self, p, q):
        # get the number of data attributes
        attributes = self.stored_X.shape[1]
        sum = 0.0
        for attribute in range(attributes):
            # take the sum of the absolute value of pi-qi to the power of P
            # where i equals values from 1 to the number of attributes(n)
            sum += np.absolute(p[attribute] - q[attribute]) ** self.P
        # then raise the sum to the power of (1/P)
        distance = sum ** (1 / self.P)
        return distance

    def cosine_distance(self, p, q):
        # cosine distance = 1 - cosine similarity
        # ---- finding cosine similarity ----
        # get the number of data attributes
        attributes = self.stored_X.shape[1]
        dot_product = 0.0
        magnitude_p = 0.0
        magnitude_q = 0.0
        # find the dot product of p and q - numerator
        # find the magnitude of p and q multiplied - denominator
        for attribute in range(attributes):
            # sum of each value squared
            magnitude_p += (p[attribute]) ** 2
            magnitude_q += (q[attribute]) ** 2
            # sum of pi times qi
            dot_product += p[attribute] * q[attribute]
        # take the square root
        magnitude_p = (magnitude_p) ** (1/2)
        magnitude_q = (magnitude_q) ** (1/2)
        # mult together
        denominator = magnitude_p * magnitude_q
        cosine_similarity = dot_product / denominator
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    def distance(self, x):
        # Calculate distances of training data to a single input data point (distances from self.stored_X to x)
        # Output np.array([distances to x])
        rows = self.stored_X.shape[0]
        distance_to_x = []
        if self.metric == "minkowski":
            # search through the rows of the stored data
            for row in range(rows):
                # take the values in the row of the dataframe and make them a list
                q = self.stored_X.loc[row].values.flatten().tolist()
                # use the distance formula on x and every row of the stored data
                # add that to a list
                distance_to_x.append(self.minkowski_distance(x, q))
            # create an array of the distances
            distances = np.asarray(distance_to_x)

        elif self.metric == "euclidean":
            # euclidean is just minkowski distance where P equals 2
            self.P = 2
            # search through the rows of the stored data
            for row in range(rows):
                # take the values in the row of the dataframe and make them a list
                b = self.stored_X.loc[row].values.flatten().tolist()
                # use the distance formula on x and every row of the stored data
                # add that to a list
                distance_to_x.append(self.minkowski_distance(x, b))
            # create an array of the distances
            distances = np.asarray(distance_to_x)

        elif self.metric == "manhattan":
            # manhattan is just minkowski distance where P equals 1
            self.P = 1
            # search through the rows of the stored data
            for row in range(rows):
                # take the values in the row of the dataframe and make them a list
                b = self.stored_X.loc[row].values.flatten().tolist()
                # use the distance formula on x and every row of the stored data
                # add that to a list
                distance_to_x.append(self.minkowski_distance(x, b))
            # create an array of the distances
            distances = np.asarray(distance_to_x)

        elif self.metric == "cosine":
            # search through the rows of the stored data
            for row in range(rows):
                # take the values in the row of the dataframe and make them a list
                b = self.stored_X.loc[row].values.flatten().tolist()
                # use the distance formula on x and every row of the stored data
                # add that to a list
                distance_to_x.append(self.cosine_distance(x, b))
            # create an array of the distances
            distances = np.asarray(distance_to_x)
        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_nearest_neighbor(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        # get the array
        distances = self.distance(x)
        # sort the array by distances and show the n nearest
        k_nearest_data = distances.argsort()[:self.n_neighbors]
        k_nearest = []
        # take the index of the shortest distances
        for index in k_nearest_data:
            # find that classification and store it in the list
            k_nearest.append(self.stored_y[index])
        # counter to 'vote' to use in the majority vote later
        vote = Counter(k_nearest)
        return vote

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.stored_X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_nearest_neighbor(x)
            # Calculate the probability of data point x belonging to each class
            # e.g. prob = {"2": 1/3, "1": 2/3}
            prob = {k: neighbors[k] / float(self.n_neighbors) for k in self.classifications}
            probs.append(prob)
        probs = pd.DataFrame(probs, columns=self.classifications)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classifications[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions