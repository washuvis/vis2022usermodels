import numpy as np
import pandas as pd
import scipy.special as sp
import math
import itertools
from util import flatten_list


class CompetingModels:
    """
    Competing Models: A Bayesian model selection approach for detecting exploration bias and inferring information relevance during
    visual exploratory analysis
    
    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
    """
    def __init__(self, data, continuous_attributes, discrete_attributes):
        self.underlying_data = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes

        attr_bool_enumeration = [[bool(x) for x in i] for i in itertools.product([0, 1], repeat=len(continuous_attributes) + len(discrete_attributes))]
        self.models_enumerated = {f'competing_model__{"___".join(flatten_list(np.array(self.continuous_attributes + self.discrete_attributes, dtype=object)[b]))}': {
            'continuous_attributes': flatten_list(np.array(continuous_attributes, dtype=object)[b[:len(continuous_attributes)]]),
            'discrete_attributes': list(np.array(discrete_attributes)[b[len(continuous_attributes):]])
        } for b in attr_bool_enumeration}
        print(f'{len(self.models_enumerated)} competing models enumerated.')

        self.continuous_model = MultivariateGaussianModel(self.underlying_data, 1, len(flatten_list(continuous_attributes))+1, continuous_attributes)
        self.discrete_models = {d_attribute: CategoricalModel(self.underlying_data, d_attribute, 1) for d_attribute in discrete_attributes}

        # for null models
        self.discrete_raw_counts = {d_attr: self.underlying_data[d_attr].value_counts().to_dict() for d_attr in discrete_attributes}

        self.model_posterior = pd.DataFrame(columns=list(self.models_enumerated.keys()))
        self.model_posterior = self.model_posterior.append({name: 1/len(self.model_posterior.columns) for name in self.model_posterior.columns}, ignore_index=True)

    def update(self, observation_index):
        """
        Update the models.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data 
        """
        # update model posterior
        last_posterior = self.model_posterior.iloc[-1]
        updated_posterior = {}
        for model in self.models_enumerated.keys():
            prob = np.log(1)

            # "important" attributes
            if len(self.models_enumerated[model]['continuous_attributes']) != 0:
                prob += np.log(self.continuous_model.predict(observation_index, self.models_enumerated[model]['continuous_attributes']))

            for d_attribute in self.models_enumerated[model]['discrete_attributes']:
                prob += np.log(self.discrete_models[d_attribute].predict(observation_index))

            # null attributes
            for null_d_attribute in np.setdiff1d(self.discrete_attributes, self.models_enumerated[model]['discrete_attributes']):
                #prob += np.log(self.discrete_raw_counts[null_d_attribute][self.underlying_data.iloc[observation_index][null_d_attribute]] / len(self.underlying_data))
                prob += np.log(1 / len(self.underlying_data))
                # prob += np.log(1 / len(self.underlying_data[null_d_attribute].unique()))

            if len(self.models_enumerated[model]['continuous_attributes']) == 0:
                df_temp = self.underlying_data.copy()
                for d_attribute in self.models_enumerated[model]['discrete_attributes']:
                    df_temp = df_temp[df_temp[d_attribute] == self.underlying_data.iloc[observation_index][d_attribute]]
                prob += np.log(1 / len(df_temp))

            # previous observations and prior
            updated_posterior[model] = np.log(last_posterior[model]) + prob

        self.model_posterior = self.model_posterior.append(updated_posterior, ignore_index=True)
        # self.model_posterior = self.model_posterior.fillna(method='ffill')
        # self.model_posterior.iloc[-1] = self.model_posterior.iloc[-1] - self.model_posterior.iloc[-1].min()
        self.model_posterior.iloc[-1] = np.exp(self.model_posterior.iloc[-1].astype(float))
        self.model_posterior.iloc[-1] = self.model_posterior.iloc[-1] / self.model_posterior.iloc[-1].sum()

        # update models in light of observation
        self.continuous_model.update(observation_index)
        for i, d_attribute in enumerate(self.discrete_attributes):
            self.discrete_models[d_attribute].update(self.underlying_data.iloc[observation_index][d_attribute])

    def predict(self):
        """
        Prediction step where the probability of each point being the next interaction is calculated

        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        prob_df = pd.DataFrame()
        prob = np.zeros(len(self.underlying_data))

        for model in self.models_enumerated.keys():
            # "important" attributes
            if len(self.models_enumerated[model]['continuous_attributes']) != 0:
                func = np.vectorize(self.continuous_model.predict, excluded={1})
                prob += np.log(func(self.underlying_data.index.values, self.models_enumerated[model]['continuous_attributes'] + ['interaction_timestamp']))

            for d_attribute in self.models_enumerated[model]['discrete_attributes']:
                func = np.vectorize(self.discrete_models[d_attribute].predict)
                prob += np.log(func(self.underlying_data.index.values))

            # null attributes
            for null_d_attribute in np.setdiff1d(self.discrete_attributes, self.models_enumerated[model]['discrete_attributes']):
                #func = np.vectorize(lambda observation_index: self.discrete_raw_counts[null_d_attribute][self.underlying_data.iloc[observation_index][null_d_attribute]] / len(self.underlying_data))
                func = np.vectorize(lambda observation_index: 1 / len(self.underlying_data))
                prob += np.log(func(self.underlying_data.index.values))
                # prob += np.log(1 / len(self.underlying_data[null_d_attribute].unique()))

            if len(self.models_enumerated[model]['continuous_attributes']) == 0:
                def func(observation_index):
                    df_temp = self.underlying_data.copy()
                    #for d_attribute in self.models_enumerated[model]['discrete_attributes']:
                    #    df_temp = df_temp[df_temp[d_attribute] == self.underlying_data.iloc[observation_index][d_attribute]]
                    return 1 / len(df_temp)
                v_func = np.vectorize(func)
                prob += np.log(v_func(self.underlying_data.index.values))
            prob_df[model] = prob

        posterior = self.get_model_posterior().iloc[-1]
        prob_df = prob_df + np.log(posterior.astype(float))
        # prob_df = prob_df - prob_df.min()
        prob_df['probability'] = np.exp(prob_df).sum(axis=1)
        return prob_df['probability']

    def get_model_posterior(self):
        """
        Pr(M | D) for every M

        Returns:
            The model's posterior
        """
        return self.model_posterior

    def get_attribute_bias(self):
        """
        Pr( biased towards each attribute | D) for every attribute

        Returns:
            A Pandas Dataframe of biases
        """
        biases = pd.DataFrame()
        for attribute in self.continuous_attributes + self.discrete_attributes:
            attribute_name = attribute
            if type(attribute) == list or type(attribute) == np.ndarray:
                attribute_name = "___".join(attribute)
            if attribute_name != 'interaction_timestamp':
                cols = [c for c in self.get_model_posterior().columns if attribute_name in c]
                biases[attribute_name] = self.get_model_posterior()[cols].sum(axis=1)
        return biases


class MultivariateGaussianModel:
    """
    According to "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin Murphy
    Section 8: Normal-Wishart Prior
    """
    def __init__(self, data, k, v, continuous_attributes, name='unspecified_name'):
        self.k_0 = k
        self.v_0 = v
        self.df_0 = v - len(flatten_list(continuous_attributes)) + 1
        self.data = data[flatten_list(continuous_attributes)].copy()
        self.data_w_prob = data[flatten_list(continuous_attributes)].copy()
        self.observations = pd.DataFrame(columns=flatten_list(continuous_attributes) + ['interaction_timestamp'])

        # mu_0 is the prior mean.
        # mean of underlying data is our prior mean.
        self.mu_0 = self.data[flatten_list(continuous_attributes)].mean().to_numpy()

        # add an attribute for time step
        self.mu_0 = np.append(self.mu_0, 0)

        # T_0 is the prior covariance.
        # for the starting covariance, we make an (n+1)x(n+1) matrix of zeros (extra dimension for time) and fill it below.
        self.T_0 = np.zeros((len(flatten_list(continuous_attributes)) + 1, len(flatten_list(continuous_attributes)) + 1))

        # add time covariance
        self.T_0[-1, -1] = 1

        # add underlying data covariance
        data_covariance = self.data[flatten_list(continuous_attributes)].cov().to_numpy() / v
        self.T_0[:-1, :-1] = data_covariance

        self.mu = self.mu_0
        self.T = self.T_0
        self.df = self.df_0
        self.k = self.k_0
        self.v = self.v_0

        attributes = self.observations.columns.to_list()
        attribute_to_index = {attributes[i]: i for i in range(len(attributes))}
        continuous_attributes.append('interaction_timestamp')
        continuous_attributes = np.array(continuous_attributes, dtype=object)
        subsets_bool = [[bool(x) for x in i] for i in itertools.product([0, 1], repeat=len(continuous_attributes)) if (sum(i) != 0) and not (sum(i) == 1 and int(i[-1]) == 1)]
        self.continuous_model_names = {'model___' + '___'.join(flatten_list(continuous_attributes[b])): {'attr_list': flatten_list(continuous_attributes[b])} for b in subsets_bool}
        for model_name in self.continuous_model_names.keys():
            self.continuous_model_names[model_name]['row_ind'] = [[attribute_to_index[a]] for a in self.continuous_model_names[model_name]['attr_list']]
            self.continuous_model_names[model_name]['col_ind'] = [attribute_to_index[a] for a in self.continuous_model_names[model_name]['attr_list']]
            self.data_w_prob[model_name] = np.vectorize(self.compute_probability)(list(range(len(self.data))), model_name)
            self.data_w_prob[model_name] = self.data_w_prob[model_name] / self.data_w_prob[model_name].sum()

    def update(self, observation):
        """
        @param observation is an index (integer)
        """
        # add the observation to list
        self.observations = self.observations.append({**self.data[self.observations.columns[:-1]].iloc[observation], 'interaction_timestamp': len(self.observations) + 1}, ignore_index=True)

        number_of_observations = len(self.observations)
        # update the model if more than one observation has arrived
        if number_of_observations > 1:
            x_bar = self.observations.mean().to_numpy()
            S = (number_of_observations - 1) * self.observations.astype(float).cov().to_numpy()
            self.T = self.T_0 + S + ((self.k_0 * number_of_observations) / (self.k_0 + number_of_observations)) * np.dot((self.mu_0 - x_bar).reshape(-1, 1), (self.mu_0 - x_bar).reshape(1, -1))
            self.mu = (self.k_0 * self.mu_0 + number_of_observations * x_bar) / (self.k_0 + number_of_observations)
            self.v = self.v_0 + number_of_observations
            self.k = self.k_0 + number_of_observations
            self.df = self.v - len(self.mu_0) + 1

        # model updated; now compute the pdf and normalize to get pmf across data points
        for model_name in self.continuous_model_names.keys():
            self.data_w_prob[model_name] = np.vectorize(self.compute_probability)(self.data.index.values, model_name)
            self.data_w_prob[model_name] = self.data_w_prob[model_name] / self.data_w_prob[model_name].sum()

    def compute_probability(self, x_index, model_name):
        # print(f'working on {x_index}, {model_name}')
        attr_list = self.continuous_model_names[model_name]['attr_list']
        row_ind = self.continuous_model_names[model_name]['row_ind']
        col_ind = self.continuous_model_names[model_name]['col_ind']

        d = len(self.mu_0)
        lamb = ((self.k + 1) / (self.k * (self.v - d + 1))) * self.T
        x = self.data.iloc[x_index].to_numpy()
        x = np.append(x, len(self.observations) + 1)
        return multivariate_t_pdf(x[col_ind], self.v, self.mu[col_ind], lamb[row_ind, col_ind])

    def predict(self, x, attributes):
        model_name = 'model___' + '___'.join(flatten_list(attributes))
        return self.data_w_prob[model_name].iloc[x]


class CategoricalModel:
    def __init__(self, data, var_name, alpha):
        self.m = 0
        self.var_name = var_name
        self.underlying_data = data
        self.categories = list(data[var_name].unique())
        self.category_to_index = {self.categories[i]: i for i in range(len(self.categories))}

        # hyper-parameter of categorical distribution
        self.alpha = alpha
        self.m = np.zeros(len(self.categories))
        self.mu = (self.alpha + self.m) / (np.sum(self.alpha + self.m))
        # print('Dirichlet model created for:', var_name)

    def update(self, observation, debug=False):
        self.m[self.category_to_index[observation]] += 1
        self.mu = (self.alpha + self.m) / (np.sum(self.alpha + self.m))

        if debug:
            print(self.var_name, "model updated; number of observations: ", len(self.observations))

    def predict(self, observation_index):
        observed_cat = self.underlying_data.iloc[observation_index][self.var_name]
        return self.mu[self.category_to_index[observed_cat]]/len(self.underlying_data[self.underlying_data[self.var_name] == observed_cat])


def multivariate_t_pdf_old(x, v, mu, lamb):
    """
    Multivariate t distribution, according to Eq. 2.162 in Bishop's PRML
    :param x: value of interest
    :param v: hyper-parameter
    :param mu: mean, center parameter
    :param lamb: Lambda, precision matrix; spread parameter
    :return: pdf value St(x | mu, lambda, nu)
    """
    # number of dimensions in x
    d = len(x)

    if v == 0:
        print('warning: v=0; changed to v=1')
        v = 1

    # final formula is (a/b)*c
    a = sp.gamma((v + d) / 2.0) * np.linalg.det(lamb) ** (1 / 2.0)
    b = sp.gamma(v / 2.0) * (v ** (d / 2.0)) * (math.pi ** (d / 2.0))
    c = (1 + ((1.0 / v) * float(np.dot((x - mu), np.dot(lamb, (x - mu)))))) ** (-(v + d) / 2.0)

    ans = (a / b) * c
    return ans


def multivariate_t_pdf(x, df, mu, sigma):
    d = len(x)
    '''
    print('x: ', x)
    print('df: ', df)
    print('mu: ', mu)
    print('sigma: ', sigma)
    '''

    # final formula is (a/b)*c
    a = sp.gamma((df + d) / 2.0)
    b = sp.gamma(df / 2.0) * df ** (d / 2.0) * math.pi ** (d / 2.0) * np.linalg.det(sigma) ** (1 / 2.0)

    c = (1 + (1.0 / df) * np.dot(np.transpose(x - mu), np.linalg.solve(sigma, (x - mu)))) ** (-(df + d) / 2.0)

    ans = (a / b) * c

    return ans


