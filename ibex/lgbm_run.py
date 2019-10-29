import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

seed = 0
asset = 'TEF'

# LGBM params
num_leaves = 16
min_child_samples = 128
max_depth = 4
learning_rate = 0.002
n_estimators = 5000
subsample = 0.35
subsample_freq = 5
colsample_bytree = 0.85

# ===== Preparing data =====
data = pd.read_csv(f'data/IBEX_{asset}_dataset.csv')
n_investors = np.unique(data.investor_encoding.values).shape[0]

train_idx = data['train_idx'].values
val_idx = data['val_idx'].values
test_idx = data['test_idx'].values

features = list(data.columns[3:-3])  # Removing irrelevant columns - date, encoding, target & splits.
data = data[['date', 'investor_encoding', 'buyer'] + features]
train_data = data[train_idx]
val_data = data[val_idx]
test_data = data[test_idx]

# ===== Training model =====
print(f'Training on {train_data.shape[0]} samples, validating on {val_data.shape[0]} samples.')
model = lgbm.LGBMClassifier(num_leaves=num_leaves, min_child_samples=min_child_samples, max_depth=max_depth,
                            learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                            subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, random_state=seed,
                            n_jobs=1)
model.fit(train_data[features + ['investor_encoding']].values, train_data.buyer.values,
          eval_set=(val_data[features + ['investor_encoding']].values, val_data.buyer.values),
          feature_name=features + ['investor_encoding'], categorical_feature=['investor_encoding'],
          early_stopping_rounds=50)

train_pred = model.predict_proba(train_data[features + ['investor_encoding']].values)
val_pred   = model.predict_proba(val_data[features + ['investor_encoding']].values)
test_pred  = model.predict_proba(test_data[features + ['investor_encoding']].values)

train_score = average_precision_score(train_data.buyer.values, train_pred[:, 1])
val_score = average_precision_score(val_data.buyer.values, val_pred[:, 1])
test_score = average_precision_score(test_data.buyer.values, test_pred[:, 1])
print(f'train ap: {100*train_score:.2f} - val ap: {100*val_score:.2f} - test ap: {100*test_score:.2f}')
