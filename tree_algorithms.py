# =============================================================================
# Decision Trees
# =============================================================================
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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
print(X_train.shape[0])
print(X_test.shape[0])

# Fit and predict
tree = DecisionTreeRegressor(max_depth=3, random_state=SEED)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print('R2 score on testing data: {:.2f}%'.format(100*r2_score(y_test, y_pred)))

# =============================================================================
# Decision Trees: Visualize 
# =============================================================================
# remember to $ pip install dtreeviz
from dtreeviz.trees import dtreeviz
viz = dtreeviz(tree, X_train, y_train, target_name='PRICE',
               feature_names=features,
               X = X_test[5,:]) # select a random testing sample for viz
viz.save("boston.svg") # suffix determines the generated image format
viz.view()             # pop up window to display image

# =============================================================================
# Random forest
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

# Fit and predict
rf = RandomForestRegressor(random_state=SEED)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('R2 score on testing data: {:.2f}%'.format(100*r2_score(y_test, y_pred)))

# Feature selection
feat_imp = rf.feature_importances_

# =============================================================================
# Gradient Boosting Machines
# =============================================================================
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import time

# Initialize instances of GBM models
gbm_models = [GradientBoostingRegressor(random_state=SEED), 
              LGBMRegressor(random_state=SEED), 
              XGBRegressor(random_state=SEED)]

# Loop over models
for gbm in gbm_models:
    # Measure time
    start_time = time.time()
    # Fit and predict
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    # Print info
    print('Model: {} \n R2 score on testing data: {:.2f}% \n Execution time: {:.2}sec'.format(
          gbm.__class__.__name__, 100*r2_score(y_test, y_pred), time.time()-start_time))