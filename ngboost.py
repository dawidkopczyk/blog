import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.ensemble import RandomForestRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import r2_score

# Load data
dataset = load_boston()
X, y = dataset.data, dataset.target
features = dataset.feature_names

SEED = 2019

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=SEED)
print('The shape of training dataset: {}'.format(X_train.shape[0]))
print('The shape of testing dataset: {}'.format(X_test.shape[0]))

# Fit and predict
rf = RandomForestRegressor(n_estimators=400, random_state=SEED).fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Random Forest: R2 score on testing data: {:.2f}%'.format(100 * r2_score(y_test, y_pred)))

# Fit and predict
lgb = LGBMRegressor(n_estimators=400, random_state=SEED).fit(X_train, y_train)
y_pred = lgb.predict(X_test)
print('LightGBM: R2 score on testing data: {:.2f}%'.format(100 * r2_score(y_test, y_pred)))

# Fit and predict
np.random.seed(SEED)
ngb = NGBRegressor(n_estimators=400,
                   Base=default_tree_learner, Dist=Normal, Score=MLE).fit(X_train, y_train)
y_pred = ngb.predict(X_test)
print('NGBoost: R2 score on testing data: {:.2f}%'.format(100 * r2_score(y_test, y_pred)))

# Probability distribution
obs_idx = [0,1]
dist = ngb.pred_dist(X_test[obs_idx, :])
print('P(y_0|x_0) is normally distributed with loc={:.2f} and scale={:.2f}'.format(dist.loc[0], dist.scale[0]))
print('P(y_1|x_1) is normally distributed with loc={:.2f} and scale={:.2f}'.format(dist.loc[1], dist.scale[1]))