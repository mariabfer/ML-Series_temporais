from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

# @MSG: ltm não, lstm

def softmax(z):
        """
        Applies softmax with no-overflows.

        params:
        - z: a array of values

        returns:
        a array of probabilities given by softmax of <code>z</code> values.
        """

        # It's subtract from max to avoid float overflow.
        z = np.asarray(z)
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)


class NodeWrapper():
    """
    A tree node wrapper containing characteristic vector and target distribution of node.
    """

    def __init__(self, C, target_distribution):
        """"
        Creates a new NodeWrapper.

        Params:
        - C: the characteristic vector of node.
        - distribution: the target distribution.
        """
        self.C = C
        self.distribution = target_distribution



class CharacteristicModel(ABC):
    """
    A model for compressing a matrix of features into a single vector of features.
    """

    @abstractmethod
    def apply(self, X):
        """
        Compresses the matrix of features into a single vector of mean of the features.

        params:
        - X: a numpy matrix of features (shape = (n_instances, n_features)) of all instances in a tree node.

        returns:
        a single vector of some function of features
        """

        # TODO: check this shape, i mean it's incorrect
        pass


class PredictModel(ABC):
    def __init__(self, decision_path_method: Literal["all", "best"]):
        self.decision_path_method = decision_path_method

    def predict(self, y_lags, probs, decision_path, target_bins, expectation_method):
        """
        Predict the expected value of target.

        params:
        - y_lags (shape(1, l)): the instance of short-term memory of the time series considered as predictors with l lags.
        - probs: the probability vector of similarity between the new predicting instance and node characteristic vector over all nodes in decision path.
        - decision_path: a list of NodeWrapper's of the decision path in tree ordered by root to leaf.
        - target_bins: the bins of the target according to the preprocessing (in the case of the i-th bin: target_bins[i][0]: left bound, target_bins[i][1]: right bound)
        - method: the expectation method. (1) "**mid**": uses the midpoint of the target_bins interval to realizes dot product with relative frequence of distribution. (2) "**stm**": uses the short-term memory of y_lags to select identifies probables intervals of target. In sequence, normalizes the relative frequence considering intervals with zero observations as impossible. Finally, applies the dot product between the normalized relative frequence and the mean of past relative to interval. (3) "**ltm**": simmilarly to "stm", but, does not realizes the normalization process, instead, when a interval does not is observed, uses the midpoint as expected value of it.
        
        returns:
        a single scalar value corresponding to the expected value for the specified entries.
        """
        depth_score = self._calculate_depth_score(probs)

        if self.decision_path_method == "best":
            most_probable_node_index = np.argmax(depth_score)
            most_probable_node = decision_path[most_probable_node_index]
            most_probable_distribution = most_probable_node.distribution

            return expected_value(target_bins, y_lags, most_probable_distribution, expectation_method)

        elif self.decision_path_method == "all":
            depth_score = softmax(depth_score)
            out = 0
            for i, node in enumerate(decision_path):
                out += depth_score[i] * expected_value(target_bins, y_lags, node.distribution, expectation_method)

            return out
        
        else:
            raise Exception("Invalid decision path method. Check documentation for more information.")

    @abstractmethod
    def _calculate_depth_score(self, probs):
        pass

class TreeWrapper(ABC):
    """
    A tree wrapper for PADT models.
    """

    @abstractmethod
    def get_decision_path(self, x):
        """
        Returns a list of nodes of decision path ordered by depth in ascending order.

        params:
        - x: a instance features vector.

        returns:
        a list of NodeWrapper's from root to leaf of decision path in tree.
        """
        pass

    @abstractmethod
    def fit(self, X, y, characteristic_model, n_categories):
        """
        Fit the overfitted tree using specified characteristic model.
        
        params:
        - X: the matrix of features.
        - y: the target.
        - characteristic_model: the characteristic model.
        - n_categories: number of bins of target.

        returns: self
        """
        pass

def get_bins_edges_from_quantiles(target, n_bins):
    """
    Returns a array of scalars corresponding to bins edges based on quantiles.

    params:
    - target: the target to be discretized.
    - n_bins: the number of expected bins to be created (this can be different if bins edges contains repeated values).
    """
    # TODO: remove repeated values of bins_edges
    target = np.asarray(target)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(target, quantiles)
    return bin_edges

def bin_target(target, n_bins: int):
    """
    Binarizes the target based on number of bins.

    params:
    - target: the target series.
    - n_bins: the number of expected bins to be created (this can be different if bins edges contains repeated values).

    returns: the target binarized based on <code>n_bins</code>, the <code>bins_edges</code> e the <code>target_bins</code>.
    """
    # TODO: returns the number of reals bins
    bins_edges = get_bins_edges_from_quantiles(target, n_bins=n_bins)
    y_binned = np.digitize(target, bins_edges[1:-1], right=True)

    target_bins = [
        (bins_edges[i], bins_edges[i+1])
        for i in range(len(bins_edges) - 1)
    ]

    return y_binned, bins_edges, target_bins


class BasePADT(BaseEstimator):
    """
    A base PADT model.
    """
    def __init__(
        self,
        tree: TreeWrapper,
        characteristic_method: CharacteristicModel,
        prediction_method: PredictModel,
        n_bins: int,
        expectation_method: Literal["mid", "ltm", "stm", "midstm", "midltm"] = "mid"
    ):
        """
        Creates a new PADT model.

        params:
        - tree: the tree wrapper model.
        - characteristic_method: the characteristic method.
        - prediction_method: the prediction method.
        - n_bins: number of expected bins of target (can be less if bins_edges of quantiles contains repeated values).
        - expectation_method: the expectation_method (check documentation for more informations).
        """        
        self.tree = tree
        self.characteristic_method = characteristic_method
        self.prediction_method = prediction_method
        self.n_bins = n_bins
        self.expectation_method = expectation_method
    
    def fit(self, X, y):
        """
        Fits the PADT.

        params:
        - X: the lags features.
        - y: the target.

        returns: self.
        """
        
        X = np.asarray(X)
        y = np.asarray(y)

        y_binned, self._bins_edges, self._target_bins = bin_target(y, self.n_bins)

        self.tree.fit(X, y_binned, self.characteristic_method, self.n_bins)

        return self

    def predict(self, X):
        """
        Predicts with PADT model.

        params:
        - X: the matrix (shape = (n_instances, n_features)) of lags.

        returns: a array (shape = (n_instances,)) of expected value.
        """
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
                scores[depth] = np.dot(node.C, x_row)
            scores /= np.sqrt(len(x_row))

            probs = softmax(scores)

            # Predict value using specified model.
            out[row] = self.prediction_method.predict(x_row, probs, decision_path, self._target_bins, self.expectation_method)

        return out


class MeanCharacteristic(CharacteristicModel):
    """
    Applies compression using mean of features.
    """
    def apply(self, X):
        return np.mean(X, axis=0)


class MedianCharacteristic(CharacteristicModel):
    """
    Applies compression using median of features.
    """

    def apply(self, X):
        return np.median(X, axis=0)


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

    def mid_term_memory():
        ns = np.zeros(n_bins)
        for y in y_lags:
            for i in range(n_bins):
                if y < target_bins[i][1] or i == n_bins - 1:
                    ns[i] += 1
                    break
        for i in range(n_bins):
            out[i] = (target_bins[i][0] + target_bins[i][1]) / 2.0
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

    elif method == "midstm":
        ns = mid_term_memory()
        for i in range(n_bins):
            if ns[i] == 0:
                rel_distribution[i] = 0
        rel_distribution /= np.sum(rel_distribution)
    
    elif method == "midltm":
        mid_term_memory()
    else:
        raise Exception()        

    return np.dot(out, rel_distribution)


def s_star(probs, alpha, beta):
    """
    The S* function.

    returns: S*(probs, alpha, beta).
    """
    depths = np.arange(0, len(probs))
    return probs * alpha - beta * depths


class LinearPredictor(PredictModel):
    def __init__(self, alpha: float, beta: float, decision_path_method: Literal["all", "best"]):
        super().__init__(decision_path_method)
        self.alpha = alpha
        self.beta = beta

    def _calculate_depth_score(self, probs):
        return np.maximum(0, s_star(probs, self.alpha, self.beta))
    

class SigmoidPredictor(PredictModel):
    def __init__(self, alpha: float, beta: float, decision_path_method: Literal["all", "best"]):
        super().__init__(decision_path_method)
        self.alpha = alpha
        self.beta = beta

    def _calculate_depth_score(self, probs):
        depth_score = 1 - 1.0 / (1 + np.exp(s_star(probs, self.alpha, self.beta)))
        return depth_score

class ExpPredictor(PredictModel):
    def __init__(self, alpha: float, beta: float, decision_path_method: Literal["all", "best"]):
        super().__init__(decision_path_method)
        self.alpha = alpha
        self.beta = beta

    def _calculate_depth_score(self, probs):
        return np.exp(s_star(probs, self.alpha, self.beta))

class TanhPredictor(PredictModel):
    def __init__(self, alpha: float, beta: float, decision_path_method: Literal["all", "best"]):
        super().__init__(decision_path_method)
        self.alpha = alpha
        self.beta = beta
        
    def _calculate_depth_score(self, probs):
        return (1 + np.tanh(s_star(probs, self.alpha, self.beta))) / 2.0


class SKlearnTreeWrapper(TreeWrapper):
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
    
    def fit(self, X, y, characteristic_model: CharacteristicModel, n_categories):
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
                if node_id not in nodes_xinstances.keys():
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
        for node_id in nodes_xinstances.keys():
            x_instances = nodes_xinstances[node_id]
            y_instances = nodes_yinstances[node_id]
            C = characteristic_model.apply(x_instances)
            self.nodes[node_id] = NodeWrapper(C, self._distribution(y_instances, n_categories))

    def _distribution(self, y_instances, n_categories):
        """
        Get the distribution of y_instances with at least <code>n_categories</code> array.
        """
        y_instances = np.asarray(y_instances, dtype=int)
        return np.bincount(y_instances, minlength=n_categories)


class PADT(BaseEstimator):
    """
    The PADT model.
    """

    def __init__(
        self,
        characteristic_method: Literal["mean", "median"] | None = None,
        prediction_method: Literal["exp", "linear", "sigmoid", "tanh"] | None = None,
        n_bins: int | None = None,
        max_tree_depth: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        decision_path_method: Literal["all", "best"] = "best",
        expectation_method: Literal["mid", "stm", "ltm", "midstm", "midltm"] = "mid"
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
        - decision_path_method: the method for selects nodes in decision path.
        - expectation_method: the method for calculated expected value of target.
        """
        self.characteristic_method = characteristic_method
        self.n_bins = n_bins
        self.max_tree_depth = max_tree_depth
        self.prediction_method = prediction_method
        self.decision_path_method = decision_path_method
        self.alpha = alpha
        self.beta = beta
        self.expectation_method = expectation_method
    
    def fit(self, X, y):
        """
        Fits the PADT model.

        params:
        - X: a matrix (shape = (n_instances, n_features)) of lags of the target series.
        - y: the target series.

        returns: self.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self._tree = SKlearnTreeWrapper(self.max_tree_depth)

        # Get characteristic method
        if self.characteristic_method == "mean":
            characteristic = MeanCharacteristic()
        elif self.characteristic_method == "median":
            characteristic = MedianCharacteristic()
        else:
            raise Exception("Unexpected characteristic. Check the documentation.")

        # Get prediction method
        if self.prediction_method == "linear":
            prediction = LinearPredictor(self.alpha, self.beta, self.decision_path_method)
        elif self.prediction_method == "exp":
            prediction = ExpPredictor(self.alpha, self.beta, self.decision_path_method)
        elif self.prediction_method == "sigmoid":
            prediction = SigmoidPredictor(self.alpha, self.beta, self.decision_path_method)
        elif self.prediction_method == "tanh":
            prediction = TanhPredictor(self.alpha, self.beta, self.decision_path_method)
        else:
            raise Exception("Unexpected predictor. Check the documentation.")

        self._padt = BasePADT(
            tree=SKlearnTreeWrapper(self.max_tree_depth),
            characteristic_method=characteristic,
            prediction_method=prediction,
            n_bins=self.n_bins,
            expectation_method=self.expectation_method
        )
        self._padt.fit(X, y)

        return self       

    def predict(self, X):
        """
        Predict using PADT.

        params:
        - X: a matrix (shape = (n_instances, n_features)) of lags of the target series.

        returns: a array (shape = (n_instances,)) of predictions for target series based on X.
        """
        return self._padt.predict(X)
