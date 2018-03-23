import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.datasets import load_boston
from sklearn.model_selection import (cross_val_score, train_test_split, 
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import r2_score

from lightgbm.sklearn import LGBMRegressor

#==============================================================================
# Load data
#==============================================================================
boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=2018)

print(X_train.shape, ' train samples shape')
print(X_test.shape, ' test samples shape')

#==============================================================================
# Grid Search CV
#==============================================================================
# Define space of hyperparameters
hyper_space = {'n_estimators': [1000, 1500, 2000, 2500],
               'max_depth':  [4, 5, 8, -1],
               'num_leaves': [15, 31, 63, 127],
               'subsample': [0.6, 0.7, 0.8, 1.0],
               'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}

# Initilize estimator
est = LGBMRegressor(boosting='gbdt', n_jobs=-1, random_state=2018)

# Grid Search CV
gs = GridSearchCV(est, hyper_space, scoring='r2', cv=4, verbose=1)
gs_results = gs.fit(X_train, y_train)
print("BEST PARAMETERS: " + str(gs_results.best_params_))
print("BEST CV SCORE: " + str(gs_results.best_score_))

# Predict (after fitting GridSearchCV is an estimator with best parameters)
y_pred = gs.predict(X_test)

# Score
score = r2_score(y_test, y_pred)
print("R2 SCORE ON TEST DATA: {}".format(score))

#==============================================================================
# Random Search CV
#==============================================================================
hyper_space = {'n_estimators': sp_randint(1000, 2500),
               'max_depth':  [4, 5, 8, -1],
               'num_leaves': [15, 31, 63, 127],
               'subsample': sp_uniform(0.6, 0.4),
               'colsample_bytree': sp_uniform(0.6, 0.4)}

# Random Search CV
rs = RandomizedSearchCV(est, hyper_space, n_iter=60, scoring='r2', cv=4, 
                         verbose=1, random_state=2018)
rs_results = rs.fit(X_train, y_train)
print("BEST PARAMETERS: " + str(rs_results.best_params_))
print("BEST CV SCORE: " + str(rs_results.best_score_))

# Predict (after fitting RandomizedSearchCV is an estimator with best parameters)
y_pred = rs.predict(X_test)

# Score
score = r2_score(y_test, y_pred)
print("R2 SCORE ON TEST DATA: {}".format(score))

#==============================================================================
# Tree-structured Parzen Estimator
#==============================================================================
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval

# Define searched space
hyper_space = {'n_estimators': 1000 + hp.randint('n_estimators', 1500),
               'max_depth':  hp.choice('max_depth', [4, 5, 8, -1]),
               'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127]),
               'subsample': hp.uniform('subsample', 0.6, 1.0),
               'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)}

# Objective function (CV R2)
def evaluate(params, X, y):
    
    # Initilize instance of estimator
    est = LGBMRegressor(boosting='gbdt', n_jobs=-1, random_state=2018)
    
    # Set params
    est.set_params(**params)
    
    # Calc CV score
    scores = cross_val_score(estimator=est, X=X, y=y, 
                             scoring='r2', cv=4)
    score = np.mean(scores)

    return score

# Objective minizmied 
hyperopt_objective = lambda params: (-1.0) * evaluate(params, X_train, y_train)

#==============================================================================
# Hyperoptimization (Tree Parzen Estimator)
#============================================================================== 
# Trail
trials = Trials()

# Set algoritm parameters
algo = partial(tpe.suggest, 
               n_startup_jobs=20, gamma=0.25, n_EI_candidates=24)

# Fit Tree Parzen Estimator
best_vals = fmin(hyperopt_objective, space=hyper_space,
                 algo=algo, max_evals=60, trials=trials,
                 rstate=np.random.RandomState(seed=2018))

# Print best parameters
best_params = space_eval(hyper_space, best_vals)
print("BEST PARAMETERS: " + str(best_params))

# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))

# Print execution time
tdiff = trials.trials[-1]['book_time'] - trials.trials[0]['book_time']
print("ELAPSED TIME: " + str(tdiff.total_seconds() / 60))    

# Set params
est.set_params(**best_params)

# Fit    
est.fit(X_train, y_train)
y_pred = est.predict(X_test)

# Predict
score = r2_score(y_test, y_pred)
print("R2 SCORE ON TEST DATA: {}".format(score))

#==============================================================================
# Tree structure of hyperparameter space (Optional)
#============================================================================== 
# You must change the evalute function in order to extract learning rate 
# and n_estimators
hyper_space_tree = {'choices': hp.choice('choices', [
                                {'learning_rate': 0.01, 
                                 'n_estimators': hp.randint('n_estimators', 1500)},
                                {'learning_rate': 0.1, 
                                 'n_estimators': 1000 + hp.randint('n_estimators', 1500)}
                                ]),
                    'max_depth':  hp.choice('max_depth', [4, 5, 8, -1]),
                    'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127]),
                    'subsample': hp.uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)}


