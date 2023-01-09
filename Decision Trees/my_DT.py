import pandas as pd
import numpy as np
from collections import Counter

##### Changed hint file
class my_DT:

    def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)
        self.tree = {}

    def impurity(self, labels):
        # Calculate impurity
        # Input is a list of labels
        # Output impurity score <= 1
        stats = Counter(labels)
        N = float(len(labels))
        if self.criterion == "gini":
            gini = 0
            for label in stats:
                gini += (stats[label] / N) ** 2
            impure = 1 - gini
        elif self.criterion == "entropy":
            entropy = 0
            for label in stats:
                entropy += (stats[label] / N) * (np.log2(stats[label] / N))
            impure = (-1) * entropy
        else:
            raise Exception("Unknown criterion.")

        return impure

    def find_best_split(self, pop, X, labels):
        # Find the best split
        # Inputs:
        #   pop:    indices of data in the node
        #   X:      independent variables of training data
        #   labels: dependent variables of training data
        # Output: tuple(best feature to split, weighted impurity score of best split, splitting point of the feature,
        #               [indices of data in left node, indices of data in right node],
        #               [weighted impurity score of left node, weighted impurity score of right node])
        ######################
        # initialize variables for output
        best_feature = None
        best_weighted_impurity = None
        best_splitting_point = None
        indices_left_node = None
        indices_right_node = None
        best_indices_left_node = None
        best_indices_right_node = None
        weighted_impurity = None
        weighted_impurity_left_right = None
        # a place-holder for the most impure
        impure = 10000
        # search through the columns of the data
        for feature in X.keys():
            cans = np.array(X[feature][pop])
            index_sorted_cans = cans.argsort()
            weighted_impurity_left_right = []
            weighted_impurity = []
            for index in range(len(index_sorted_cans) - 1):
                # split the data into left right
                indices_left_node = pop[index_sorted_cans[:index + 1]]
                indices_right_node = pop[index_sorted_cans[index + 1:]]
                # if the values aren't the same
                if cans[index_sorted_cans[index]] != cans[index_sorted_cans[index + 1]]:
                    # calc impurity for the left and right
                    # weighted impurity scores for each node (# data in node * unweighted impurity)
                    weighted_impurity_left_right.append([self.impurity(labels[indices_left_node])
                                                         * (len(indices_left_node)),
                                                         self.impurity(labels[indices_right_node])
                                                         * len(indices_right_node)])
                else:
                    # add a place-holder that will not impact
                    weighted_impurity_left_right.append([impure, impure])
                # either way add the impurity
                weighted_impurity.append(np.sum(weighted_impurity_left_right[-1]))
            # get the best total impurity
            best_weighted_impurity = np.amin(weighted_impurity)
            # first 2 constraints are for the fist time around,
            # the last is checking whether the new 'best' is better than the last 'best'
            if best_weighted_impurity < impure and (best_feature == None or best_weighted_impurity < best_feature[1]):
                # find the index of the most pure
                index_splitting_point = np.argmin(weighted_impurity)
                # find that point and the one after and use the middle
                best_splitting_point = (cans[index_sorted_cans][index_splitting_point]
                                        + cans[index_sorted_cans][index_splitting_point + 1]) / 2.0
                # use that to find the 'best' left and right node indices
                best_indices_left_node = pop[index_sorted_cans[:index_splitting_point + 1]]
                best_indices_right_node = pop[index_sorted_cans[index_splitting_point + 1:]]
                # store the 'best' values
                best_feature = (feature, best_weighted_impurity, best_splitting_point,
                                [best_indices_left_node, best_indices_right_node],
                                weighted_impurity_left_right[index_splitting_point])

        return best_feature

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        labels = np.array(y)
        N = len(y)
        population = {0: np.array(range(N))}
        impurity = {0: self.impurity(labels[population[0]]) * N}
        level = 0
        nodes = [0]
        while level < self.max_depth and nodes:
            next_nodes = []
            for node in nodes:
                current_pop = population[node]
                current_impure = impurity[node]
                if len(current_pop) < self.min_samples_split or current_impure == 0 or level + 1 == self.max_depth:
                    self.tree[node] = Counter(labels[current_pop])
                else:
                    best_feature = self.find_best_split(current_pop, X, labels)
                    if best_feature and (current_impure - best_feature[1]) > self.min_impurity_decrease * N:
                        self.tree[node] = (best_feature[0], best_feature[2])
                        next_nodes.extend([node * 2 + 1, node * 2 + 2])
                        population[node * 2 + 1] = best_feature[3][0]
                        population[node * 2 + 2] = best_feature[3][1]
                        impurity[node * 2 + 1] = best_feature[4][0]
                        impurity[node * 2 + 2] = best_feature[4][1]
                    else:
                        self.tree[node] = Counter(labels[current_pop])
            nodes = next_nodes
            level += 1

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list

        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    label = list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
                    predictions.append(label)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # Example:
        # self.classes_ = {"2", "1"}
        # the reached node for the test data point has {"1":2, "2":1}
        # then the prob for that data point is {"2": 1/3, "1": 2/3}
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)

        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    # Calculate prediction probabilities for data point arriving at the leaf node.
                    # predictions = list of prob, e.g. prob = {"2": 1/3, "1": 2/3}
                    # the N we left out (denominator) need the sum for none left node solutions
                    N = float(np.sum(list(self.tree[node].values())))
                    # divide each count by N and place in the for above
                    predictions.append({class_: self.tree[node][class_] / N for class_ in self.classes_})
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        probs = pd.DataFrame(predictions, columns=self.classes_)

        return probs
