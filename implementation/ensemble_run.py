"""
This script runs all modeling techniques (for data relevance) for a given dataset and user.
It saves one pandas dataframe per timestamp in the given session.
"""

import sys
from pathlib import Path
import ast
sys.path.append('../implementation')

import warnings
warnings.filterwarnings('ignore')
# TODO look into these warnings and address them

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.special as sp
import time

from util import flatten_list
from ottley_hidden_markov_model import HMM
from monadjemi_competing_models import CompetingModels
from zhou_analytic_focus import AnalyticFocusModel
from healey_adaboost_naive_bayes import AdaBoostNB
from weighted_k_nearest_neighbors import WeightedKNN
from wall_bias import Wall
from gotz_adaptive_contextualization import AC

underlying_data_paths = {
    'stl': '../data/stl_crimes/dots.csv',
    'vast': '../data/vast_2011_challenge/vast_data_sample_reduced.pkl',
    'boardrooms': '../data/boardrooms/boardrooms_data.csv',
    'political': '../data/political/final/political.csv'
}

user_interaction_data_paths = {
    'stl': '../data/stl_crimes/stl_combined_interactions.csv',
    'vast': '../data/vast_2011_challenge/bookmark_interactions_clean.pkl',
    'boardrooms': '../data/boardrooms/boardrooms_combined_interactions.csv',
    'political': '../data/political/final/wall_political_interactions.csv'
}

continuous_attributes = {
    'stl': [['x', 'y']],
    'vast': [['latitude', 'longitude']],
    'boardrooms': ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay'],
    'political': ['age', 'political_experience', 'policy_strength_ban_abortion_after_6_weeks',
                  'policy_strength_legalize_medical_marijuana', 'policy_strength_increase_medicare_funding',
                  'policy_strength_ban_alcohol_sales_sundays']
}

discrete_attributes = {
    'stl': ['type'],
    'vast': ['topic'],
    'boardrooms': ['industry'],
    'political': ['party', 'gender', 'occupation']
}

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Not the right number of arguments are given.')
        print('Usage: python3 ensemble_run [dataset] [session] [bias or rel]')
        sys.exit(2)

    dataset = sys.argv[1].lower()
    session = int(sys.argv[2])
    modeling = sys.argv[3].lower()
    if modeling == 'bias':
        dir_name = dataset + "_bias"
        output_dir_path = f'../output/ensembles/{dir_name}'
    else:
        output_dir_path = f'../output/ensembles/{dataset}/{session}'
    print(f'Running all models on user {session} of dataset {dataset}')

    # load and prepare underlying data
    print('Loading and preparing data')
    underlying_data = None
    interaction_data = None

    if dataset == 'stl':
        underlying_data = pd.read_csv(underlying_data_paths[dataset])
        underlying_data.set_index('id', drop=True, inplace=True)
        underlying_data['type'] = underlying_data['type'] - 1

        interaction_data = pd.read_csv(user_interaction_data_paths[dataset])
        interaction_data['interaction_session'] = interaction_data.apply(lambda row: ast.literal_eval(row.interaction_session), axis=1)
        interaction_data['interaction_type_session'] = interaction_data.apply(lambda row: ast.literal_eval(row.interaction_type_session), axis=1)

    elif dataset == 'vast':
        underlying_data = pd.read_pickle(underlying_data_paths[dataset])
        for d_attr in discrete_attributes[dataset]:
            underlying_data[d_attr] = pd.factorize(underlying_data[d_attr])[0]

        interaction_data = pd.read_pickle(user_interaction_data_paths[dataset])
        interaction_data = interaction_data[interaction_data['experimental_group'] == 'control']
        interaction_data = interaction_data.reset_index(drop=True)

    elif dataset == 'boardrooms':
        underlying_data = pd.read_csv(underlying_data_paths[dataset])
        underlying_data = underlying_data[discrete_attributes[dataset] + flatten_list(continuous_attributes[dataset])].copy()

        for d_attr in discrete_attributes[dataset]:
            underlying_data[d_attr] = pd.factorize(underlying_data[d_attr])[0]

        interaction_data = pd.read_csv(user_interaction_data_paths[dataset])
        interaction_data['interaction_session'] = interaction_data.apply(lambda row: ast.literal_eval(row.interaction_session), axis=1)

    elif dataset == 'political':
        underlying_data = pd.read_csv(underlying_data_paths[dataset])
        clean_id = [s.replace("p", "") for s in underlying_data['id']]
        clean_id = [int(s.lstrip('0')) - 1 for s in clean_id]
        underlying_data['id'] = clean_id
        underlying_data = underlying_data.set_index('id')
        underlying_data = underlying_data.sort_index()
        for d_attr in discrete_attributes[dataset]:
            underlying_data[d_attr] = pd.factorize(underlying_data[d_attr])[0]

        interaction_data = pd.read_csv(user_interaction_data_paths[dataset])
        interaction_data['interaction_session'] = interaction_data.apply(lambda row: ast.literal_eval(row.interaction_session), axis=1)
        interaction_data['interaction_type'] = interaction_data.apply(lambda row: ast.literal_eval(row.interaction_type), axis=1)

    else:
        print(f'Invalid dataset detected: {dataset}')
        sys.exit(2)

    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    print(f'Storing outputs in directory {output_dir_path}')

    # initiate models
    print('Initiating Models')
    hmm = HMM(underlying_data,
              continuous_attributes[dataset].copy(),
              discrete_attributes[dataset].copy(),
              1000)

    cm = CompetingModels(underlying_data,
                         continuous_attributes[dataset].copy(),
                         discrete_attributes[dataset].copy())

    if modeling == 'rel':
        af = AnalyticFocusModel(underlying_data,
                            continuous_attributes[dataset].copy(),
                            discrete_attributes[dataset].copy())

        bnb = AdaBoostNB(underlying_data,
                        continuous_attributes[dataset].copy(),
                        discrete_attributes[dataset].copy())

        knn = WeightedKNN(underlying_data,
                        continuous_attributes[dataset].copy(),
                        discrete_attributes[dataset].copy(),
                        k=20)
        
        for i in tqdm(range(len(interaction_data.iloc[session].interaction_session))):
            probs = np.zeros((len(underlying_data), 5))
            interaction = interaction_data.iloc[session].interaction_session[i]
            for m_ind, m in enumerate([hmm, cm, af, bnb, knn]):
                m.update(interaction)
                probs[:, m_ind] = m.predict()

            with open(f'{output_dir_path}/{i}.npy', 'wb+') as f:
                np.save(f, probs)
                
    elif modeling == 'bias':
        ad = Wall(underlying_data,
                  continuous_attributes[dataset].copy(),
                  discrete_attributes[dataset].copy())
        ac = AC(underlying_data,
                continuous_attributes[dataset].copy(),
                discrete_attributes[dataset].copy())
        probs = []
        print(len(interaction_data.iloc[session].interaction_session))
        for i in tqdm(range(len(interaction_data.iloc[session].interaction_session))):
            interaction = interaction_data.iloc[session].interaction_session[i]
            for m_ind, m in enumerate([hmm, cm, ad, ac]):
                m.update(interaction)
                if i == len(interaction_data.iloc[session].interaction_session) - 1:
                    bias_df = m.get_attribute_bias()
                    bias_df.columns = bias_df.columns.str.replace('bias_', '')
                    bias_df.columns = bias_df.columns.str.replace('_disc', '')
                    if dataset == 'stl':
                        bias_df['mixed'] = bias_df['x___y'] * bias_df['type']
                    bias_df.columns = ['bias-' + str(col) for col in bias_df.columns]
                    probs.append(bias_df)
        avg_df = pd.DataFrame()
        for i in range(len(probs)):
            if i == 0:
                avg_df = probs[0]
            if i < len(probs) - 2:
                avg_df = avg_df.add(probs[i+1][:len(interaction_data.iloc[session].interaction_session)], fill_value=0)
        ensemble_df = avg_df/4
        ensemble_df.to_pickle(f'{output_dir_path}/{session}.pkl')
