import sys
sys.path.append('../implementation')
import numpy as np
import collections
import pandas as pd
from util import flatten_list, hellinger_dist, discretize_auto

class AC:
    """
    Adaptive Contextualization: A metric to measure selection bias with the Helligener distance.

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
    """
    def __init__(self, data, continuous_attributes, discrete_attributes):
        self.data = data
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.interactions = pd.DataFrame()
        self.biases = {}
        self.num_interaction = 0
        self.combined_attributes = {}

        # discretize the continuous variables
        for attr in continuous_attributes[::-1]:
            if type(attr) != str:
                attr_name = '___'.join(attr)
                self.combined_attributes[attr_name] = [f'{x}_disc' for x in attr]
                for cont_attr in flatten_list(attr):
                    discretize_auto(self.data, cont_attr, 3)
                self.discrete_attributes = np.append(f'{attr_name}', self.discrete_attributes)
            else:
                discretize_auto(self.data, attr, 3)
                self.discrete_attributes = np.append(f'{attr}_disc', self.discrete_attributes)
        
    def update(self, interaction_data):
        """
        Takes in a user's interaction point and calculates the Helligener distance between the observed
        distribution and the underlying data distribution for each attribute. Stores the distance as the attribute's bias.

        Parameters:
            interaction_data: A Pandas Dataframe of an observed interaction
        """
        self.num_interaction += 1
        # adding in the new interaction
        self.interactions = self.interactions.append(self.data.loc[interaction_data])
        # looping through the discrete attributes
        for attr in self.discrete_attributes:
            all_probs = []
            if attr in self.combined_attributes:
                for x in self.combined_attributes[attr]:
                    # finding all the unique elements for attr
                    categories = np.unique(self.data[x])

                    # proportions in full data
                    l = len(self.data[x])
                    c = collections.Counter(self.data[x])
                    proportions = np.array([c[t] / l for t in categories])

                    # calculating the observed frequencies
                    c_ui = collections.Counter(self.interactions[x])
                    observed_freq = np.array([c_ui[t] if t in c_ui.keys() else 0 for t in categories])/self.num_interaction
                    dist = hellinger_dist(observed_freq, proportions)
                    all_probs.append(dist)
            else:
                # finding all the unique elements for attr
                categories = np.unique(self.data[attr])

                # proportions in full data
                l = len(self.data[attr])
                c = collections.Counter(self.data[attr])
                proportions = np.array([c[t] / l for t in categories])

                # calculating the observed frequencies
                c_ui = collections.Counter(self.interactions[attr])
                observed_freq = np.array([c_ui[t] if t in c_ui.keys() else 0 for t in categories])/self.num_interaction
                dist = hellinger_dist(observed_freq, proportions)
                all_probs.append(dist)
                
            final_dist = np.prod(all_probs)
            if attr in self.biases:
                self.biases[attr].append(final_dist)
            else:
                self.biases[attr] = [final_dist]

    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return pd.DataFrame(self.biases)