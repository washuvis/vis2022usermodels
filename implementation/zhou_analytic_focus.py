import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from util import flatten_list, discretize_auto


class AnalyticFocusModel:
    """
    Analytic Focus: A method that tracks user focus on each of the concepts by observing their interactions and maintain an importance score for each concept.
    
    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
    """
    def __init__(self, data, continuous_attributes, discrete_attributes):
        self.underlying_data = data
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.interaction_indices = np.array([], dtype=int)
        self.normalizer = MinMaxScaler()

        # discretize the continuous attributes
        for attr in flatten_list(continuous_attributes):
            discretize_auto(self.underlying_data, attr, 3)
            self.discrete_attributes = np.append(self.discrete_attributes, f'{attr}_disc')
        self.underlying_data_w_probability = data.copy()

        # values used in paper; in the more general case, these depend on interaction type
        self.persistence = 2
        self.initial_importance = 1

        # enumerate concepts for every attribute (refer to the paper for the definition of a "concept")
        self.concepts = {attr: self.underlying_data[attr].unique() for attr in self.discrete_attributes}
        self.concepts_importance = {attr: {c: 1 for c in self.concepts[attr]} for attr in self.discrete_attributes}
        self.observation_timestamps = {attr: {c: np.array([]) for c in self.concepts[attr]} for attr in self.discrete_attributes}
        self.observation_persistence = {attr: {c: np.array([]) for c in self.concepts[attr]} for attr in self.discrete_attributes}
        self.observation_initial_importance = {attr: {c: np.array([]) for c in self.concepts[attr]} for attr in self.discrete_attributes}

    def update(self, observation_index):
        """
        Update the model.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data 
        """
        self.interaction_indices = np.append(self.interaction_indices, observation_index)
        for attr in self.discrete_attributes:
            this_attr_obv_value = self.underlying_data.iloc[observation_index][attr]
            for concept in self.concepts[attr]:
                # decay past observations
                self.observation_timestamps[attr][concept] = self.observation_timestamps[attr][concept] - 1

                if this_attr_obv_value == concept:
                    # 0 stands for the current timestamp
                    self.observation_timestamps[attr][concept] = np.append(self.observation_timestamps[attr][concept], 0)
                    self.observation_persistence[attr][concept] = np.append(self.observation_persistence[attr][concept], self.persistence)
                    self.observation_initial_importance[attr][concept] = np.append(self.observation_initial_importance[attr][concept], self.initial_importance)

                # update concept importance (Eq. 1 from paper)
                self.concepts_importance[attr][concept] = (self.observation_initial_importance[attr][concept]*(np.exp(self.observation_timestamps[attr][concept]/self.observation_persistence[attr][concept]))).sum()

    def predict(self):
        """
        Prediction step where the importance of each point being the next interaction is calculated

        Returns:
            A Pandas Dataframe of the underlying data with importances that represent certainity that the point is the next interaction point
        """
        importance = np.ones(len(self.underlying_data))
        for attr in self.discrete_attributes:
            self.underlying_data_w_probability[f'{attr}__importance'] = self.underlying_data_w_probability[attr]
            self.underlying_data_w_probability[f'{attr}__importance'].replace(self.concepts_importance[attr], inplace=True)
            importance *= self.underlying_data_w_probability[f'{attr}__importance'].to_numpy()

        self.underlying_data_w_probability['importance'] = importance
        return self.underlying_data_w_probability['importance']

    def get_attribute_bias(self):
        return
