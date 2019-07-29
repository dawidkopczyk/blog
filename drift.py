# Import objects
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.datasets import load_boston

# Load data
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Randomly split into train and test datasets
df_train, df_test = train_test_split(df, random_state=1)

# Add ID variable (we expect that the variable will be idetified as drifiting)
df_train['ID'] = np.arange(len(df_train))
df_test['ID'] = np.arange(len(df_train),len(df))

# Loop over each column in X and calculate ROC-AUC
drifts = []
for col in df_train.columns:
    # Select column
    X_train = df_train[[col]]
    X_test = df_test[[col]]

    # Add origin feature
    X_train["target"] = 0
    X_test["target"] = 1

    # Merge datasets
    X_tmp = pd.concat((X_train, X_test),
                      ignore_index=True).drop(['target'], axis=1)
    y_tmp= pd.concat((X_train.target, X_test.target),
                   ignore_index=True)

    X_train_tmp, X_test_tmp, \
    y_train_tmp, y_test_tmp = train_test_split(X_tmp,
                                               y_tmp,
                                               test_size=0.25,
                                               random_state=1)

    # Use Random Forest classifier
    rf = RandomForestClassifier(n_estimators=50,
                                n_jobs=-1,
                                max_features=1.,
                                min_samples_leaf=5,
                                max_depth=5,
                                random_state=1)

    # Fit
    rf.fit(X_train_tmp, y_train_tmp)

    # Predict
    y_pred_tmp = rf.predict_proba(X_test_tmp)[:, 1]

    # Calculate ROC-AUC
    score = roc_auc_score(y_test_tmp, y_pred_tmp)

    drifts.append((max(np.mean(score), 1 - np.mean(score)) - 0.5) * 2)

print(drifts)