# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
import numpy as np
from util import flatten_list, discretize_auto


class AdaBoostNB():
    """
    Boosted-Naive Bayes Classifer: A naive Bayes classifier for maintaining a belief over users' interest in data points.

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
    """
    def __init__(self, data, continuous_attributes, discrete_attributes):
        self.data = data.copy()
        self.data_with_probability = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.interactions = pd.DataFrame()
        self.observation_labels = np.array([])
        self.min_categories = []

        # discretizing the continuous attributes
        for attr in flatten_list(continuous_attributes):
            discretize_auto(self.data, attr, 3)
            self.discrete_attributes = np.append(self.discrete_attributes, f'{attr}_disc')

        self.data = self.data[self.discrete_attributes]

        # adding in random negative point in the beginning
        self.interactions = self.interactions.append(self.data.sample())
        self.observation_labels = np.append(self.observation_labels, 0)

        # getting unique number of categories for each attribute
        for attr in self.data.columns:
            self.min_categories.append(len(self.data[attr].unique()))

        # create base classifier
        cnb = CategoricalNB(class_prior=[0.9, 0.1], min_categories=self.min_categories)

        # Create adaboost classifier object
        self.abc = AdaBoostClassifier(base_estimator=cnb)

    def update(self, interaction_index):
        """
        Takes in an interaction index (the id of a data point in the underlying data) and adds into observed interactions.
        Fits the Naive Bayes Classifer with the observed interaction labels.

        Parameterss:
            interaction_index: An integer that represents an id of a data point in the underlying data 
        """
        interaction = self.data.iloc[interaction_index]
        self.interactions = self.interactions.append(interaction)
        self.observation_labels = np.append(self.observation_labels, 1)
        self.abc.fit(self.interactions, self.observation_labels)

    def predict(self):
        """
        Predicting the next interaction point.

        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        probabilities = self.abc.predict_proba(self.data)[:, 1]
        self.data_with_probability['probability'] = probabilities
        return self.data_with_probability['probability']

