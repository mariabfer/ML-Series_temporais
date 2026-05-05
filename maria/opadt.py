from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

class OnlineNodeWrapper():
    def __init__(self, n, C, target_distribution):
        """"
        Creates a new NodeWrapper.

        Params:
        - C: a characteristic vector.
        - n: the number of instances reaches the node in training. 
        - distribution: the target bin distribution.
        """
        self.n = n
        self.C = C
        self.distribution = target_distribution

    def update(self, x, y):
        self.C = (self.C * self.n + x) / (self.n + 1)
        self.n += 1
        self.distribution[y] += 1


class OnlinePredictModel(ABC):
    @abstractmethod
    def predict_one(self, y_lags, probs, decision_path, target_bins, expectation_method):
        """
        Predict the expected value of target.

        params:
        - y_lags (shape(1, l)): the instance of short-term memory of the time series considered as predictors with l lags.
        - probs: the probability vector of similarity between the new predicting instance and node characteristic vector over all nodes in decision path.
        - decision_path: a list of NodeWrapper's of the decision path in tree ordered by root to leaf.
        - target_bins: the bins of the target according to the preprocessing (in the case of the i-th bin: target_bins[i][0]: left bound, target_bins[i][1]: right bound)
        - method: the expectation method. (1) "**mid**": uses the midpoint of the target_bins interval to realizes dot product with relative frequence of distribution. (2) "**stm**": uses the short-term memory of y_lags to select identifies probables intervals of target. In sequence, normalizes the relative frequence considering intervals with zero observations as impossible. Finally, applies the dot product between the normalized relative frequence and the mean of past relative to interval. (3) "**ltm**": simmilarly to "stm", but, does not realizes the normalization process, instead, when a interval does not is observed, uses the midpoint as expected value of it.
        """
        pass

class OnlineTreeWrapper(ABC):
    @abstractmethod
    def get_decision_path(self, x) -> list[OnlineNodeWrapper]:
        """
        Returns a list of nodes of decision path ordered by depth in ascending order.

        Params:
        - x: a instance features vector.
        """
        pass

    @abstractmethod
    def fit(self, X, y, n_categories):
        """
        Fit the overfitted tree using specified characteristic model.
        
        Params:
        - X: the matrix of features.
        - y: the target.
        - characteristic_model: the characteristic model.
        - n_categories: number of bins of target.
        """
        pass


def get_bins_edges_from_quantiles(x, n_bins):
    x = np.asarray(x)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(x, quantiles)
    return bin_edges

def bin_target(y_series: pd.Series, n_bins: int):
    bins_edges = get_bins_edges_from_quantiles(y_series, n_bins=n_bins)
    y_binned = np.digitize(y_series, bins_edges[1:-1], right=True)

    target_bins = [
        (bins_edges[i], bins_edges[i+1])
        for i in range(len(bins_edges) - 1)
    ]

    return y_binned, bins_edges, target_bins


class BaseOnlinePADT(BaseEstimator):
    def __init__(
        self,
        tree: OnlineTreeWrapper,
        prediction_method: OnlinePredictModel,
        n_bins: int,
        expectation_method: Literal["mid", "ltm", "stm"] = "mid"
    ):
        self.tree = tree
        self.prediction_method = prediction_method
        self.n_bins = n_bins
        self.expectation_method = expectation_method
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        y_binned, self._bins_edges, self._target_bins = bin_target(y, self.n_bins)

        self.tree.fit(X, y_binned, self.n_bins)

        return self
    
    def predict(self, X):
        X = np.asarray(X)
        n_rows = X.shape[0]
        out = np.empty(n_rows)

        for row in range(n_rows):
            x_row = X[row]
            
            # Get decision path in the tree.
            decision_path = self.tree.get_decision_path(x_row)
            n_nodes_in_decision_path = len(decision_path)

            # Apply attention mechanism over all characteristic vectors of path nodes.
            scores = np.empty(n_nodes_in_decision_path)
            for depth, node in enumerate(decision_path):
                scores[depth] = np.dot(node.C, x_row) # < node.C / ||node.C||, x_row / ||x_row|| >
            probs = self._softmax(scores)

            # Predict value using specified model.
            out[row] = self.prediction_method.predict_one(x_row, probs, decision_path, self._target_bins, self.expectation_method)

        return out


    def fit_update(self, x, y):
        x = np.asarray(x)        
        y_binned = np.digitize([y], self._bins_edges[1:-1], right=True)

        decision_path = self.tree.get_decision_path(x)
        for node in decision_path:
            node.update(x, y_binned)

    def predict_one(self, x):        
        x = np.asarray(x)
            
        # Get decision path in the tree.
        decision_path = self.tree.get_decision_path(x)
        n_nodes_in_decision_path = len(decision_path)

        # Apply attention mechanism over all characteristic vectors of path nodes.
        scores = np.empty(n_nodes_in_decision_path)
        for depth, node in enumerate(decision_path):
            scores[depth] = np.dot(node.C, x)
        probs = self._softmax(scores)

        # Predict value using specified model.
        return self.prediction_method.predict_one(x, probs, decision_path, self._target_bins, self.expectation_method)

    def _softmax(self, z):
        # It's subtract from max to avoid float overflow.
        z = np.asarray(z)
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)


def expected_value(target_bins, y_lags, distribution, method: Literal["mid", "stm", "ltm"] = "mid") -> float:
    """
    Get the expected value of the target.

    Params:
    - target_bins: the bins of the target according to the preprocessing (in the case of the i-th bin: target_bins[i][0]: left bound, target_bins[i][1]: right bound)
    - y_lags (shape(1, l)): the instance of short-term memory of the time series considered as predictors with l lags.
    - distribution: the target absolute frequence distribution
    - method: the expectation method. "**mid**": uses the midpoint of the target_bins interval to realizes dot product with relative frequence of distribution. "**stm**": uses the short-term memory of y_lags to select identifies probables intervals of target. In sequence, normalizes the relative frequence considering intervals with zero observations as impossible. Finally, applies the dot product between the normalized relative frequence and the mean of past relative to interval. "**ltm**": simmilarly to "stm", but, does not realizes the normalization process, instead, when a interval does not is observed, uses the midpoint as expected value of it.
    """
    n_bins = len(target_bins)
    out = np.zeros(n_bins)
    rel_distribution = np.copy(distribution + 1.0)
    rel_distribution /= np.sum(rel_distribution)

    def term_memory():
        ns = np.zeros(n_bins)
        for y in y_lags:
            for i in range(n_bins):
                if y < target_bins[i][1] or i == n_bins - 1:
                    ns[i] += 1
                    out[i] = (out[i] * (ns[i] - 1) + y) / ns[i]
                    break
        return ns

    if method == "mid":
        for i in range(n_bins):
            out[i] = (target_bins[i][0] + target_bins[i][1]) / 2.0
    elif method == "stm":
        ns = term_memory()
        for i in range(n_bins):
            if ns[i] == 0:
                rel_distribution[i] = 0
        rel_distribution /= np.sum(rel_distribution)
    elif method == "ltm":
        ns = term_memory()
        for i in range(n_bins):
            if ns[i] == 0:
                out[i] = (target_bins[i][0] + target_bins[i][1]) / 2.0
    return np.dot(out, rel_distribution)

def s_star(probs, alpha, beta):
    depths = np.arange(0, len(probs))
    return probs * alpha - beta * depths

class LinearPredictor(OnlinePredictModel):
    def __init__(self, alpha: float, beta: float):
        """
        Creates a new LinearPredictor. Predicts as y_t = E[decision_path[argmax(alpha * S - beta * d)]].
        """
        self.alpha = alpha
        self.beta = beta
    
    def predict_one(self, y_lags, probs, decision_path, target_bins, expectation_method):
        # Apply linear decay and selects the most probable node.
        depth_score = self._calculate_depth_score(probs)
        most_probable_node_index = np.argmax(depth_score)

        # Calculates expected value.
        most_probable_node = decision_path[most_probable_node_index]
        most_probable_distribution = most_probable_node.distribution
        out = expected_value(target_bins, y_lags, most_probable_distribution, expectation_method)

        return out
    
    def _calculate_depth_score(self, probs):
        return np.maximum(0, s_star(probs, self.alpha, self.beta))

class SigmoidPredictor(OnlinePredictModel):
    def __init__(self, alpha: float, beta: float):
        """
        Creates a new SigmoidPredictor. Predicts as y_t = E[decision_path[argmax(1 - 1/(1 + exp(alpha * S - beta * d)))]].
        """
        self.alpha = alpha
        self.beta = beta
    
    def predict_one(self, y_lags, probs, decision_path, target_bins, expectation_method):
        # Apply linear decay and selects the most probable node.
        depth_score = self._calculate_depth_score(probs)
        most_probable_node_index = np.argmax(depth_score)

        # Calculates expected value.
        most_probable_node = decision_path[most_probable_node_index]
        most_probable_distribution = most_probable_node.distribution
        out = expected_value(target_bins, y_lags, most_probable_distribution, expectation_method)

        return out

    def _calculate_depth_score(self, probs):
        depth_score = 1 - 1.0 / (1 + np.exp(s_star(probs, self.alpha, self.beta)))
        return depth_score

class ExpPredictor(OnlinePredictModel):
    def __init__(self, alpha: float, beta: float):
        """
        Creates a new ExpPredictor. Predicts as y_t = E[decision_path[argmax(exp(alpha * S - beta * d))]].
        """
        self.alpha = alpha
        self.beta = beta
    
    def predict_one(self, y_lags, probs, decision_path, target_bins, expectation_method):
        # Apply pow decay and selects the most probable node.
        depth_score = self._calculate_depth_score(probs)
        most_probable_node_index = np.argmax(depth_score)

        # Calculates expected value.
        most_probable_node = decision_path[most_probable_node_index]
        most_probable_distribution = most_probable_node.distribution
        out = expected_value(target_bins, y_lags, most_probable_distribution, expectation_method)

        return out

    def _calculate_depth_score(self, probs):
        return np.exp(s_star(probs, self.alpha, self.beta))

class TanhPredictor(OnlinePredictModel):
    def __init__(self, alpha: float, beta: float):
        """
        Creates a new TanhPredictor. Predicts as y_t = E[decision_path[argmax((tanh(alpha * S - beta * d) + 1) / 2)]].
        """
        self.alpha = alpha
        self.beta = beta
    
    def predict_one(self, y_lags, probs, decision_path, target_bins, expectation_method):
        # Apply pow decay and selects the most probable node.
        depth_score = self._calculate_depth_score(probs)
        most_probable_node_index = np.argmax(depth_score)

        # Calculates expected value.
        most_probable_node = decision_path[most_probable_node_index]
        most_probable_distribution = most_probable_node.distribution
        out = expected_value(target_bins, y_lags, most_probable_distribution, expectation_method)
        
        return out
    
    def _calculate_depth_score(self, probs):
        return (1 + np.tanh(s_star(probs, self.alpha, self.beta))) / 2.0


class OnlineSKlearnTreeWrapper(OnlineTreeWrapper):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def get_decision_path(self, X):
        X = np.asarray(X)
        
        t = self._tree.tree_
        out = []
        node_id = 0
        children_left = t.children_left
        children_right = t.children_right
        feature = t.feature
        threshold = t.threshold

        while True:
            out.append(self.nodes[node_id])
            if children_left[node_id] == children_right[node_id]:
                break
        
            feat = feature[node_id]
            thr = threshold[node_id]
            if X[feat] <= thr:
                node_id = children_left[node_id]
            else:
                node_id = children_right[node_id]
        
        return out
    
    def fit(self, X, y, n_categories):
        X = np.asarray(X)
        y = np.asarray(y)

        self._tree = DecisionTreeClassifier(random_state=42, max_depth=self.max_depth)
        self._tree.fit(X, y)

        nodes_xinstances = {}
        nodes_yinstances = {}
        t = self._tree.tree_
        children_left = t.children_left
        children_right = t.children_right
        feature = t.feature
        threshold = t.threshold
        for row in range(X.shape[0]):
            x_row = X[row]
            y_row = y[row]

            node_id = 0
            while True:
                if node_id not in nodes_xinstances:
                    nodes_xinstances[node_id] = []
                    nodes_yinstances[node_id] = []
                nodes_xinstances[node_id].append(x_row)
                nodes_yinstances[node_id].append(y_row)

                if children_left[node_id] == children_right[node_id]:
                    break

                feat = feature[node_id]
                thr = threshold[node_id]

                if x_row[feat] <= thr:
                    node_id = children_left[node_id]
                else:
                    node_id = children_right[node_id]

        self.nodes = {}
        for node_id in nodes_xinstances:
            x_instances = nodes_xinstances[node_id]
            y_instances = nodes_yinstances[node_id]
            C = np.mean(x_instances, axis=0)
            self.nodes[node_id] = OnlineNodeWrapper(len(x_instances), C, self._distribution(y_instances, n_categories))

    def _distribution(self, y_instances, n_categories):
        y_instances = np.asarray(y_instances, dtype=int)
        return np.bincount(y_instances, minlength=n_categories)


class OnlinePADT(BaseEstimator):
    def __init__(
        self,
        prediction_method: Literal["exp", "linear", "sigmoid", "tanh"] | None = None,
        n_bins: int | None = None,
        max_tree_depth: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        expectation_method: Literal["mid", "stm", "ltm"] = "mid"
    ):
        """
        Creates a new Path Attention Decision Tree model.

        Params:
        - characteristic_method: the characteristic method for compressing instances in a unique vector.
        - prediction_method: the depth score prediction method.
        - n_bins: number of bins of target.
        - max_tree_depth: the max depth of tree.
        - alpha: the alpha parameter for calculating depth score in predictions.
        - beta: the beta parameter for calculating depth score in predictions.
        - expectation_method: the method for calculated expected value of target.
        """
        self.n_bins = n_bins
        self.max_tree_depth = max_tree_depth
        self.prediction_method = prediction_method
        self.alpha = alpha
        self.beta = beta
        self.expectation_method = expectation_method
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self._tree = OnlineSKlearnTreeWrapper(self.max_tree_depth)

        # Get prediction method
        if self.prediction_method == "linear":
            prediction = LinearPredictor(self.alpha, self.beta)
        elif self.prediction_method == "exp":
            prediction = ExpPredictor(self.alpha, self.beta)
        elif self.prediction_method == "sigmoid":
            prediction = SigmoidPredictor(self.alpha, self.beta)
        elif self.prediction_method == "tanh":
            prediction = TanhPredictor(self.alpha, self.beta)
        else:
            raise Exception("Unexpected predictor. Check the documentation.")

        self._opadt = BaseOnlinePADT(
            tree=OnlineSKlearnTreeWrapper(self.max_tree_depth),
            prediction_method=prediction,
            n_bins=self.n_bins,
            expectation_method=self.expectation_method
        )
        self._opadt.fit(X, y)

        return self       
    
    def predict(self, X):
        return self._opadt.predict(X)

    def predict_one(self, x):
        return self._opadt.predict_one(x)

    def fit_update(self, x, y):
        self._opadt.fit_update(x, y)
