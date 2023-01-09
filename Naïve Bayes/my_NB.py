import pandas as pd
import numpy as np
from collections import Counter


class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha
        self.p_x_y = {}
        self.p_y = {}
        self.tally_y = {}
        self.classifications = {}

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # -----------------------------------------------------------------
        # list of classifications in y
        self.classifications = np.unique(y)
        # List of columns in X
        columns_x = list(X.columns)
        # Number of rows(both x and y)
        num_rows = X.shape[0]

        # ---------- Calculate P(y) - Class Prior ------------
        # P(y) -> need to count the likelihood -> = # of times classification was made/total number
        # Tally the number of times each classification has been used
        self.tally_y = Counter(y)
        self.p_y = {}
        for classification in self.tally_y.keys():
            self.p_y[classification] = self.tally_y[classification] / num_rows

        # ---------- Set up to calculate P(x|y) - Likelihood -----------
        # P(x|y) -> given y, need to find the likelihood of each attr -> = P(x1|y), P(x2|y), ... , P(xn|y)
        attributes_per_column = {}
        for column in columns_x:
            # add the column index as the key
            attributes_per_column[column] = []
            # add the unique attributes of each column to the list
            for attribute in np.unique(X[column]):
                attributes_per_column[column].append(attribute)

        self.p_x_y = {}
        for classification in self.classifications:
            # add the classification as a key and create a dictionary for column and likelihood
            self.p_x_y[classification] = {}
            for column in columns_x:
                # tally the number each attribute in the column only where the classification is
                tally_x_y = Counter(X[column][y == classification])
                # add the columns as a key and another dictionary for likelihood of each attribute
                self.p_x_y[classification][column] = {}
                for unique_attribute in attributes_per_column[column]:
                    # laplace estimate = (number of instances where the attribute occurs given y + alpha)
                    # / ((number of times y was classified) + (alpha * number of attributes))
                    self.p_x_y[classification][column][unique_attribute] = (tally_x_y[unique_attribute] + self.alpha) \
                                / ((self.tally_y[classification]) + (self.alpha * len(attributes_per_column[column])))
        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # -------------------------
        columns_x = list(X.columns)
        # dictionary to gather values for probabilities
        probs = {}
        # the dataframe to return, columns are the classifications in y
        probs_df = pd.DataFrame(columns=self.classifications)
        for row in range(X.shape[0]):
            for classification in self.classifications:
                # create a list to store the likelihood
                probs[classification] = []
                for column in columns_x:
                    # check to make sure the attribute has a value
                    if X.loc[row, column] in self.p_x_y[classification][column]:
                        # add the likelihood (in p_x_y) of the attribute to the list
                        probs[classification].append(self.p_x_y[classification][column][X.loc[row, column]])
                    # if the attribute doesn't have a value add 1 (won't change the product)
                    else:
                        probs[classification].append(1)
                # then multiply the likelihoods in the list and multiply that to P(y)
                probs[classification] = ((np.prod(probs[classification])) * self.p_y[classification])
            # add the dictionary to the corresponding row in the dataframe
            probs_df.loc[row] = probs
        # change to probabilities = p(y|x) / (p(y|x) + p(!y|x))
        # denominator is the sum of its row
        probs_df = probs_df.div(probs_df.sum(axis=1), axis=0)
        # for when there is not enough data to pick a max (when alpha = 0)
        probs_df['nan'] = 0
        return probs_df

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # --------------------------------
        # create a list of the column names for the max values in each row of probs_df
        predictions = list(self.predict_proba(X).idxmax(axis='columns'))
        return predictions