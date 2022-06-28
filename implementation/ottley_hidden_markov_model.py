import pathlib
import numpy as np
from random import seed, getstate, setstate
from re import search
import time
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from util import flatten_list, lognormpdf

eps = 2 ** -52  # Defined to match Matlab's default eps


class HMM:
    """
    Hidden Markov Model: An approach for modeling user attention during visual exploratory analysis.

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
        num_particles: An integer representing the number of particles used for diffusion
    """
    def __init__(self, data, continuous_attributes, discrete_attributes, num_particles):
        self.underlying_data = data
        self.underlying_data_w_probability = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.num_particles = num_particles

        # normalize all continuous variables to be between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
        self.underlying_data[flatten_list(self.continuous_attributes)] = min_max_scaler.fit_transform(self.underlying_data[flatten_list(self.continuous_attributes)])

        # initiate the pi vector to 0.5 for every attribute; and sigma for pi is 0.1; these values are arbitrary at the moment
        # self.bias_vector = 0.5 * np.ones(len(continuous_attributes) + len(discrete_attributes))
        self.bias_sigma = 0.1

        # initiate the sigma vector for every continuous attribute; set it to underlying data's standard deviation
        self.continuous_sigma = {attr: 0.25*self.underlying_data[attr].std() for attr in flatten_list(self.continuous_attributes)}

        # initiate the p vector for discrete attributes; p is the probability the user's attention flips to some other value of a given attribute
        self.discrete_p = {attr: 1/len(self.underlying_data[attr].unique()) for attr in self.discrete_attributes}

        # initiate particles randomly; particles are of form [c1, c2, c3, ..., d1, d2, d3, ..., pi1,pi2, pi3,...]
        # where c is a continuous attr, d is a discrete attribute, and pi is a bias value
        attribute_names = [x if type(x) == str else '___'.join(x) for x in continuous_attributes + discrete_attributes]
        self.bias_column_names = [f'bias_{x}' for x in attribute_names]
        self.bias_over_time = pd.DataFrame(columns=self.bias_column_names)

        self.particles = pd.DataFrame(columns=flatten_list(continuous_attributes) + discrete_attributes + self.bias_column_names)
        for attr in flatten_list(continuous_attributes):
            self.particles[attr] = np.random.rand(self.num_particles)
        for attr in self.bias_column_names:
            self.particles[attr] = np.random.rand(self.num_particles)
        for attr in discrete_attributes:
            unique_values = self.underlying_data[attr].unique()
            proportions = self.underlying_data[attr].value_counts(normalize=True)[unique_values]
            self.particles[attr] = np.random.choice(unique_values, p=proportions, size=self.num_particles)

    def update(self, observation_index):
        """
        Update the HMM model.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data 
        """
        # defuse particles
        self.defuse_particles()

        # get "click probabilities"; this is a |particles| x |underlying_data| matrix
        # element i, j is the probability of observing particle i given that data point j was our last observation.
        probability_matrix = self.get_probability_matrix()

        # weight particles by evidence and resample particles according to the weights
        weights = probability_matrix[:, observation_index]
        resampled_indices = np.random.choice(self.num_particles, self.num_particles, p=weights/np.sum(weights))
        self.particles = self.particles.iloc[resampled_indices]

        # record click probabilities
        data_point_probabilities = probability_matrix.mean(axis=0)
        self.underlying_data_w_probability['probability'] = data_point_probabilities

        self.bias_over_time = self.bias_over_time.append(self.particles[self.bias_column_names].mean(), ignore_index=True)

    def defuse_particles(self):
        """
        Diffusion of particles on all attributes
        """

        # diffusion on continuous attributes
        for attr in flatten_list(self.continuous_attributes):
            self.particles[attr] = self.particles[attr] + self.continuous_sigma[attr] * np.random.normal(size=self.num_particles)

        # diffusion on bias variables
        for attr in self.bias_column_names:
            self.particles[attr] = self.particles[attr] + self.bias_sigma * np.random.normal(size=self.num_particles)

        # diffusion on types ("discrete diffusion" as defined in the paper)
        for attr in self.discrete_attributes:
            flip_indices = np.random.random(self.num_particles) < self.discrete_p[attr]
            unique_values = self.underlying_data[attr].unique()
            proportions = self.underlying_data[attr].value_counts(normalize=True)[unique_values]
            self.particles.loc[flip_indices, attr] = np.random.choice(unique_values, size=np.count_nonzero(flip_indices), p=proportions)

        # clip values to boundary
        self.particles[flatten_list(self.continuous_attributes)] = self.particles[flatten_list(self.continuous_attributes)].clip(lower=eps, upper=1-eps)
        self.particles[self.bias_column_names] = self.particles[self.bias_column_names].clip(lower=eps, upper=1-eps)

    def get_probability_matrix(self):
        """
        This function returns a |particles| x |data_points| matrix of probabilities
        Element i, j is the probability of particle i given that data point j was our last observation.
        
        Returns:
            A numpy matrix
        """
        particles = self.particles
        data_points = self.underlying_data
        probability_matrix = np.zeros((len(particles), len(data_points)))

        # for continuous attributes, find the probabilities
        for attr in self.continuous_attributes:
            # handling cases where multiple continuous attributes come together with one bias (e.g. lat & lng coming together as location)
            attr_name = attr
            attr_list = attr
            if type(attr) != str:
                attr_name = '___'.join(attr)
            else:
                attr_list = [attr]

            attr_log_pdf = np.zeros((len(particles), len(data_points)))
            for continuous_attr in attr_list:
                attr_log_pdf += lognormpdf(particles[continuous_attr].to_numpy().reshape(-1, 1), data_points[continuous_attr].to_numpy(), self.continuous_sigma[continuous_attr])

            # shift for numerical stability purposes
            attr_log_pdf -= np.max(attr_log_pdf, axis=1)[..., np.newaxis]
            # print(attr_log_pdf)

            # exponential to get pdf from logpdf
            attr_log_pdf = np.exp(attr_log_pdf)

            # normalize to get pmf
            attr_log_pdf = attr_log_pdf / np.sum(attr_log_pdf, axis=1)[..., np.newaxis]

            # add the weighted probabilities according to particle bias value
            bias = particles[f'bias_{attr_name}'].to_numpy()
            probability_matrix += bias.reshape(-1, 1) * attr_log_pdf

        # for discrete attributes, find probability
        for attr in self.discrete_attributes:
            attr_probability = particles[attr].to_numpy().reshape(-1, 1) == data_points[attr].to_numpy()
            attr_probability = attr_probability / attr_probability.sum(axis=1)[..., np.newaxis]

            # add the weighted probabilities according to particle bias value
            bias = particles[f'bias_{attr}'].to_numpy()
            probability_matrix += bias.reshape(-1, 1) * attr_probability

        return probability_matrix

    def predict(self):
        """
        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        return self.underlying_data_w_probability['probability']

    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return self.bias_over_time
