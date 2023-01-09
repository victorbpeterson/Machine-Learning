import pandas as pd
import numpy as np
from copy import deepcopy

# Didn't use the hint. Used the pseudocode from slides.
class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.alpha = []

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classifications = np.unique(y)
        classes = np.array(y)
        numInstances = (y.shape[0])
        # (1) initialize the weights for all rows
        weight = np.array([1.0 / numInstances] * numInstances)
        # (2)
        numBoostingRounds = len(self.classifications)
        # (3)
        for estimate in range(self.n_estimators):
            # (4) Create training set by sampling with replacement from D with respect to the weight
            XiIndex = np.random.choice(a = numInstances, size = numInstances, p = weight)
            # (5) Train a base classifier
            Xi = X.iloc[XiIndex]
            # (6)
            self.estimators[estimate].fit(Xi, classes[XiIndex])
            predictions = self.estimators[estimate].predict(X)
            # (7) calc weighted error
            ci = (np.array(predictions) != y)
            error = np.sum(ci * weight)

            # (8) while the error
            while error >= (1 - 1.0 / len(self.classifications)):
                # (9) Reset the weights
                weight = np.array([1.0 / numInstances] * numInstances)
                # (10) Go back to step 4
                XiIndex = np.random.choice(a=numInstances, size=numInstances, p=weight)
                Xi = X.iloc[XiIndex]
                self.estimators[estimate].fit(Xi, numBoostingRounds[XiIndex])
                predictions = self.estimators[estimate].predict(X)
                ci = (np.array((predictions) != y))
                error = np.sum(ci * weight)
                # (11) end if we find a perfect prediction
                if error == 0:
                    self.alpha = [1]
                    self.estimators = [self.n_estimators[estimate]]
                    break
            # (12) Calculate alpha (Solution (K classes))
            self.alpha.append(np.log((1.0 - error)/error)) # + np.log(numInstances - 1.0))
            # (13) Update the weight of each instance
            for round in range(len(ci)):
                if ci[round] == False:
                    weight[round] = weight[round]
                else:
                    weight[round] = weight[round] * np.exp(self.alpha[-1])
            # (14) normalize
            weight = weight / np.sum(weight)
        self.alpha = self.alpha / np.sum(self.alpha)
        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = {}
        for classification in self.classifications:
            added = 0;
            for estimate in range(self.n_estimators):
                added += (self.alpha[estimate] * (self.estimators[estimate].predict(X) == classification))
            probs[classification] = added
        probs = pd.DataFrame(probs, columns = self.classifications)

        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = []
        getPred = self.predict_proba(X)
        for pred in getPred.to_numpy():
            predictions.append(self.classifications[np.argmax(pred)])
        return predictions





