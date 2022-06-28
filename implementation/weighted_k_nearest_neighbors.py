import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.metrics import log_loss, pairwise_distances
from scipy.optimize import minimize, Bounds
from scipy.sparse import csc_matrix
from util import flatten_list, eq_dist_function


class WeightedKNN:
    """
    k-Nearest Neighbors: A method for learning users' data interest by observing interaction. This approach assumes that proximity drives a user's exploration patterns. 

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
        weight_matrix_dir_path: Path to the weight matrix, default is none
        alpha: An array that holds the prior probability of relevance
        k: An integer that represents the number of neighbors used
        num_restarts: An integer that represents the number of restarts
    """
    def __init__(self, data, continuous_attributes, discrete_attributes, weight_matrix_dir_path=None, alpha=[0.9, 0.1], k=50, num_restarts=50):
        self.underlying_data = data
        self.underlying_data_w_probability = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.alpha = alpha  # prior probability of relevance;
        self.q_vector = np.ones(len(continuous_attributes) + len(discrete_attributes))
        self.weights = {}
        self.k = min(k, len(data))
        self.bias_over_time = pd.DataFrame()

        self.num_restarts = num_restarts  # for optimizing q
        self.bounds = Bounds(0, 1)  # for optimizing q

        self.interaction_indices = np.array([], dtype=int)

        # check if the neighbors' matrix exists
        if weight_matrix_dir_path is None:
            # if not compute neighborhood matrices
            for i, attr in enumerate(self.continuous_attributes):
                attribute_name = attr
                if type(attr) == list:
                    attribute_name = '___'.join(attr)
                else:
                    self.continuous_attributes[i] = [attr]
                    attr = [attr]
                print(f'Computing neighborhood matrix for {attribute_name}')
                knn_indices = np.zeros((len(self.underlying_data), self.k))
                delta = min(3000, len(self.underlying_data))
                for i in range(0, len(self.underlying_data), delta):
                    distance = pairwise_distances(self.underlying_data.iloc[i:i+delta][attr], self.underlying_data[attr])
                    for ri in range(delta):
                        for cj in range(i, i + delta):
                            if ri + i == cj:  # if on diagonal
                                distance[ri, cj] = np.inf
                    for j in range(delta):
                        sorted_indices = np.argsort(distance[j, :])[:k]
                        knn_indices[i + j, :] = sorted_indices
                rows = np.array([])
                cols = np.array([])
                for i in range(len(knn_indices)):
                    rows = np.append(rows, np.zeros(len(knn_indices[i])) + i)
                    cols = np.append(cols, knn_indices[i])
                data = np.ones(len(rows)).astype(int)
                knn_weights = csc_matrix((data, (rows.astype(int), cols.astype(int))))
                self.weights[attribute_name] = knn_weights

            for attr in self.discrete_attributes:
                print(f'Computing neighborhood matrix for {attr}')
                enc = OneHotEncoder()
                data = enc.fit_transform(self.underlying_data[attr].to_numpy().reshape(-1, 1))
                knn_indices = np.zeros((len(self.underlying_data), self.k))
                # knn_indices = {}
                delta = min(3000, len(self.underlying_data))
                for i in range(0, len(self.underlying_data), delta):
                    distance = pairwise_distances(data[i:i + delta, :], data[:, :])
                    for ri in range(delta):
                        for cj in range(i, i + delta):
                            if ri + i == cj:  # if on diagonal
                                distance[ri, cj] = np.inf
                    for j in range(delta):
                        sorted_indices = np.argsort(distance[j, :])[:k]
                        knn_indices[i + j, :] = sorted_indices
                        # indices = np.where(distance[j, :] == 0)[0]
                        # knn_indices[i+j] = indices
                rows = np.array([])
                cols = np.array([])
                for i in range(len(knn_indices)):
                    rows = np.append(rows, np.zeros(len(knn_indices[i])) + i)
                    cols = np.append(cols, knn_indices[i])
                data = np.ones(len(rows)).astype(int)
                knn_weights = csc_matrix((data, (rows.astype(int), cols.astype(int))))
                self.weights[attr] = knn_weights
        else:
            # if so, load the weight matrices
            None

    def update(self, observation_index):
        """
        Update the k-NN model.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data 
        """
        self.interaction_indices = np.append(self.interaction_indices, observation_index)
        train_ind = self.interaction_indices
        observed_labels = np.ones(len(train_ind))

        def get_loss(q):
            old_q = self.q_vector
            self.q_vector = q
            probs = self.predict()[train_ind]
            self.q = old_q
            return log_loss(observed_labels, probs, labels=[0, 1])
        '''
        min_val = float('inf')
        best_q = None

        for q0 in np.random.uniform(size=(self.num_restarts, len(self.q_vector))):
            res = minimize(
                get_loss, x0=q0, bounds=self.bounds, method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun
                best_q = res.x
        self.q_vector = best_q
        '''

        biases = {k: self.q_vector[i] for i, k in enumerate(self.weights)}
        self.bias_over_time = self.bias_over_time.append(biases, ignore_index=True)

    def predict(self):
        """
        Prediction step where the probability of each point being the next interaction is calculated

        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        train_ind = self.interaction_indices
        test_ind = np.arange(len(self.underlying_data))
        observed_labels = np.ones(len(train_ind))
        probs = {}
        total_prob = np.zeros(len(test_ind))
        for i, m in enumerate(self.weights):
            weights = self.weights[m]
            attr_probs = np.empty((len(self.underlying_data), 2))
            pos_ind = (observed_labels == 1)
            masks = [~pos_ind, pos_ind]

            for class_ in range(2):
                tmp_train_ind = train_ind[masks[class_]]
                attr_probs[:, class_] = self.alpha[class_] + (
                    weights[:, tmp_train_ind][test_ind].sum(axis=1).flatten()
                )

            attr_probs = normalize(attr_probs, axis=1, norm='l1')[:, 1]
            total_prob += self.q_vector[i] * attr_probs
            probs[m] = attr_probs

        self.underlying_data_w_probability['probability'] = total_prob / self.q_vector.sum()

        return self.underlying_data_w_probability['probability']

    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return self.bias_over_time
