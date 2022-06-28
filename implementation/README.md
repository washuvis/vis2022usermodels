# Implementations

This folder contains the standardized implementations of the seven selected user modeling techniques we compared in this paper.

- `ensemble_run.py`
    - Script to run the ensemble method.
- `gotz_adaptive_contextualization.py` (+)
    - Implementation of Adaptive Contextualization by Gotz et al.
- `healey_adaboost_naive_bayes.py` (*)
    - Implementation of Boosted Naive Bayes by Healey & Dennis
- `monadjemi_competing_models.py` (*+)
    - Implementation of Competing Models by Monadjemi et al.
- `ottley_hidden_markov_model.py` (*+)
    - Implementation of Hidden Markov Model by Ottley et al.
- `wall_bias.py` (^)
    - Implementation of Attribute Distribution by Wall et al.
- `weighted_k_nearest_neighbors.py` (*)
    - Implementation of *k*-Nearest Neighbors by Monadjemi et al.
- `zhou_analytic_focus.py` (*)
    - Implemenation of Analytic Focus by Zhou et al.

<br>
* denotes that the technique can predict next interaction <br>
+ denotes that the techchique can detect exploration bias