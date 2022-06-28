import sys
sys.path.append('../implementation')
import numpy as np
from scipy import stats
import collections
import pandas as pd
import math
from scipy.stats import ks_2samp
from util import flatten_list

class Wall:
    """
    Attribute Distribution: A metric to measure congitive bias with the Chi-Square and Kolmogorov-Smirnov test.

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
        self.chisq_assumption = {}
        
    def update(self, interaction_data):
        """
        Takes in a user's interaction point and goes through all the attributes in the underlying data.
        If the attribute is discrete, it compares the observed interaction distribution and the underlying data distribution with the Chi-Square test.
        If the attribute is continuous, it compares the observed interaction distribution and the underlying data distribution with the Kolmogorov-Smirnov test.
        The attribute's bias is calcluated by 1 - pvalue. 

        Parameters:
            interaction_data: A Pandas Dataframe of an observed interaction
        """
        # adding in the new interaction
        self.interactions = self.interactions.append(self.data.loc[interaction_data])
        
        # looping through the continuous attributes
        for attr in self.continuous_attributes:
            attr_name = attr
            attr_list = attr
            all_probs = []
            if type(attr) != str:
                attr_name = '___'.join(attr)   
                for continuous_attr in attr_list:
                    # grabbing all the interactions with attr
                    user_interaction_data = np.array(self.interactions[continuous_attr]).flatten()
                    full_data = np.array(self.data[continuous_attr]).flatten()
                    
                    # performing Kolmogorov-Smirnov (KS) test and calculate attribute distribution metric
                    ks_stat, p = ks_2samp(user_interaction_data, full_data)
                    b_Ad_attr = 1 - p
                    all_probs.append(b_Ad_attr)
            else:
                user_interaction_data = np.array(self.interactions[attr]).flatten()
                full_data = np.array(self.data[attr]).flatten()
                
                # performing Kolmogorov-Smirnov (KS) test and calculate attribute distribution metric
                ks_stat, p = ks_2samp(user_interaction_data, full_data)
                b_Ad_attr = 1 - p
                all_probs.append(b_Ad_attr)
            # check to see if bias for attr has been observed already
            final_b_Ad_attr = np.prod(b_Ad_attr)
            if attr_name in self.biases:
                self.biases[attr_name].append(final_b_Ad_attr)
            else:
                self.biases[attr_name] = [final_b_Ad_attr]
            
            if attr_name in self.chisq_assumption:
                self.chisq_assumption[attr_name].append(None)
            else:
                self.chisq_assumption[attr_name] = [None]
        
        # looping through the discrete attributes
        for attr in self.discrete_attributes:
            
            # finding all the unique elements for attr
            categories = np.unique(self.data[attr])

            # proportions in full data
            l = len(self.data[attr])
            c = collections.Counter(self.data[attr])
            proportions = np.array([c[t] / l for t in categories])

            # calculating the observed frequencies
            c_ui = collections.Counter(self.interactions[attr])
            observed_freq = np.array([c_ui[t] if t in c_ui.keys() else 0 for t in categories])

            # finding the expected frequencies
            expected_freq = np.array(
                [int(np.ceil(len(self.interactions) * proportions[list(categories).index(t)])) for t in categories])

            # check to see if the values in both the expected and observed frequencies are greater than 5
            if(all(obs > 5 for obs in observed_freq) and all(exp > 5 for exp in expected_freq)):
                if attr in self.chisq_assumption:
                    self.chisq_assumption[attr].append(True)
                else:
                    self.chisq_assumption[attr] = [True]
            else:
                if attr in self.chisq_assumption:
                    self.chisq_assumption[attr].append(False)
                else:
                    self.chisq_assumption[attr] = [False]
            # performing chi square test
            chisq, p = stats.chisquare(observed_freq, f_exp=expected_freq, ddof=-1)
            b_Ad_attr = 1 - p
            # check to see if bias for attr has been observed already
            if attr in self.biases:
                self.biases[attr].append(b_Ad_attr)
            else:
                self.biases[attr] = [b_Ad_attr]
                    
    def get_interaction_session(self):
        """
        Retrieves the interaction session

        Returns:
            A Pandas Dataframe of the observed interactions
        """
        return self.interactions
    
    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return pd.DataFrame(self.biases)
    
    def get_attribute_chi_sq(self):
        """
        Retrieves whether each attribute passes the Chi-Square test

        Returns:
            A Pandas Dataframe
        """
        return pd.DataFrame(self.chisq_assumption)
                
    