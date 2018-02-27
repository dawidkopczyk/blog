# General
import numpy as np

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Utilities
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from copy import copy as make_copy

#==============================================================================
# Generate classification data    
#==============================================================================
SEED = 2018

X, y = make_classification(n_samples=10000, n_features=40, n_redundant=0,
                           n_classes=2, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=SEED)

#==============================================================================
# Define Base (level 0) and Stacking (level 1) estimators
#==============================================================================
base_clf = [LogisticRegression(), RandomForestClassifier(), 
            AdaBoostClassifier(), SVC(probability=True)]
stck_clf = LogisticRegression()

#==============================================================================
# Evaluate Base estimators separately
#==============================================================================
for clf in base_clf:
    
    # Set seed
    if 'random_state' in clf.get_params().keys():
        clf.set_params(random_state=SEED)
    
    # Fit model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print('{} Accuracy: {:.2f}%'.format(clf.__class__.__name__, acc * 100))

#==============================================================================
# Create Hold Out predictions (meta-features)
#==============================================================================
def hold_out_predict(clf, X, y, cv):
        
    """Performing cross validation hold out predictions for stacking"""
    
    # Initilize
    n_classes = len(np.unique(y)) # Assuming that training data contains all classes
    meta_features = np.zeros((X.shape[0], n_classes)) 
    n_splits = cv.get_n_splits(X, y)
    
    # Loop over folds
    print("Starting hold out prediction with {} splits for {}.".format(n_splits, clf.__class__.__name__))
    for train_idx, hold_out_idx in cv.split(X, y): 
        
        # Split data
        X_train = X[train_idx]    
        y_train = y[train_idx]
        X_hold_out = X[hold_out_idx]
        
        # Fit estimator to K-1 parts and predict on hold out part
        est = make_copy(clf)
        est.fit(X_train, y_train)
        y_hold_out_pred = est.predict_proba(X_hold_out)
        
        # Fill in meta features
        meta_features[hold_out_idx] = y_hold_out_pred

    return meta_features

#==============================================================================
# Create meta-features for training data
#==============================================================================
# Define 4-fold CV
cv = KFold(n_splits=4, random_state=SEED)

# Loop over classifier to produce meta features
meta_train = []
for clf in base_clf:
    
    # Create hold out predictions for a classifier
    meta_train_clf = hold_out_predict(clf, X_train, y_train, cv)
    
    # Remove redundant column
    meta_train_clf = np.delete(meta_train_clf, 0, axis=1).ravel()
    
    # Gather meta training data
    meta_train.append(meta_train_clf)
    
meta_train = np.array(meta_train).T 

#==============================================================================
# Create meta-features for testing data
#==============================================================================
meta_test = []
for clf in base_clf:
    
    # Create hold out predictions for a classifier
    clf.fit(X_train, y_train)
    meta_test_clf = clf.predict_proba(X_test)
    
    # Remove redundant column
    meta_test_clf = np.delete(meta_test_clf, 0, axis=1).ravel()
    
    # Gather meta training data
    meta_test.append(meta_test_clf)
    
meta_test = np.array(meta_test).T 

#==============================================================================
# Predict on Stacking Classifier
#==============================================================================
# Set seed
if 'random_state' in stck_clf.get_params().keys():
    stck_clf.set_params(random_state=SEED)

# Optional (Add original features to meta)
original_flag = False
if original_flag:
    meta_train = np.concatenate((meta_train, X_train), axis=1)
    meta_test = np.concatenate((meta_test, X_test), axis=1)

# Fit model
stck_clf.fit(meta_train, y_train)

# Predict
y_pred = stck_clf.predict(meta_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print('Stacking {} Accuracy: {:.2f}%'.format(stck_clf.__class__.__name__, acc * 100))
