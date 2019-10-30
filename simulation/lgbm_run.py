import pandas as pd
import numpy as np
import pickle
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, accuracy_score

seed = 0
np.random.seed(seed)
mode = 'cluster'

# LGBM params
num_leaves = 64
min_child_samples = 32
max_depth = 10
learning_rate = 0.005
n_estimators = 5000
subsample = 0.25
subsample_freq = 2
colsample_bytree = 1.

data = pd.read_csv('data/sampled_data.csv')
with open('data/sampled_data_specs.pkl', 'rb') as f:
    data_specs = pickle.load(f)

features = [feature for feature in data.columns if 'X' in feature]
clients = data.Client.unique()
clients_groups = [client.split('_')[0] for client in clients]

if mode == 'classic':
    np.random.shuffle(clients)  # Clients are shuffled so as not to provide info to the model through clients' encodings
    clients_encoding = dict(zip(clients, range(len(clients))))
    data['client_encoding'] = [clients_encoding[client] for client in data.Client]
elif mode == 'cluster':
    clusters_mapping = dict(zip(np.unique(clients_groups), range(len(np.unique(clients_groups)))))
    clusters = [clusters_mapping[c] for c in clients_groups]
    clients_encoding = dict(zip(clients, clusters))
    data['client_encoding'] = [clients_encoding[client] for client in data.Client]
else:
    raise ValueError('Try mode in {classic, cluster}.')

# Splitting & shuffling data.
train_data = data.iloc[data_specs['train_split']].sample(frac=1.)
val_data = data.iloc[data_specs['valid_split']].sample(frac=1.)
test_data = data.iloc[data_specs['test_split']].sample(frac=1.)

print(f'Training on {train_data.shape[0]} samples, validating on {val_data.shape[0]} samples.')
lgbm_model = lgbm.LGBMClassifier(num_leaves=num_leaves, min_child_samples=min_child_samples, max_depth=max_depth,
                                 learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                                 subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, random_state=seed)
lgbm_model.fit(train_data[features + ['client_encoding']].values, train_data.Y.values,
               eval_set=(val_data[features + ['client_encoding']].values, val_data.Y.values),
               feature_name=features + ['client_encoding'], categorical_feature=['client_encoding'],
               early_stopping_rounds=50)

train_pred = lgbm_model.predict_proba(train_data[features + ['client_encoding']].values)
val_pred   = lgbm_model.predict_proba(val_data[features + ['client_encoding']].values)
test_pred  = lgbm_model.predict_proba(test_data[features + ['client_encoding']].values)

# ===== Classification results =====
train_acc = accuracy_score(train_data.Y.values, train_pred[:, 1] > .5)
val_acc = accuracy_score(val_data.Y.values, val_pred[:, 1] > .5)
test_acc = accuracy_score(test_data.Y.values, test_pred[:, 1] > .5)

train_auc = roc_auc_score(train_data.Y.values, train_pred[:, 1])
val_auc = roc_auc_score(val_data.Y.values, val_pred[:, 1])
test_auc = roc_auc_score(test_data.Y.values, test_pred[:, 1])

p_train_acc = accuracy_score(train_data.Y.values, train_data.proba.values > .5)
p_val_acc = accuracy_score(val_data.Y.values, val_data.proba.values > .5)
p_test_acc = accuracy_score(test_data.Y.values, test_data.proba.values > .5)

p_train_auc = roc_auc_score(train_data.Y.values, train_data.proba.values)
p_val_auc = roc_auc_score(val_data.Y.values, val_data.proba.values)
p_test_auc = roc_auc_score(test_data.Y.values, test_data.proba.values)

print('----- overall results -----')
print(f'train acc: {100*train_acc:.2f} - val acc: {100*val_acc:.2f} - test acc: {100*test_acc:.2f}')
print(f'train auc: {100*train_auc:.2f} - val auc: {100*val_auc:.2f} - test auc: {100*test_auc:.2f}')
print(f'(perfect) train acc: {100*p_train_acc:.2f} - val acc: {100*p_val_acc:.2f} - test acc: {100*p_test_acc:.2f}')
print(f'(perfect) train auc: {100*p_train_auc:.2f} - val auc: {100*p_val_auc:.2f} - test auc: {100*p_test_auc:.2f}')

print('\n----- per cluster test results -----')
for cluster in ['active', 'common', 'inactive']:
    cluster_idx = [cluster in x for x in test_data.Client.values]
    cluster_pred = lgbm_model.predict_proba(test_data.iloc[cluster_idx][features + ['client_encoding']].values)[:, 1]
    cluster_acc = accuracy_score(test_data.iloc[cluster_idx].Y.values, cluster_pred > .5)
    cluster_auc = roc_auc_score(test_data.iloc[cluster_idx].Y.values, cluster_pred)
    p_cluster_acc = accuracy_score(test_data.iloc[cluster_idx].Y.values, test_data.iloc[cluster_idx].proba.values > .5)
    p_cluster_auc = roc_auc_score(test_data.iloc[cluster_idx].Y.values, test_data.iloc[cluster_idx].proba.values)
    print(f'{cluster} test acc: {100*cluster_acc:.2f} - {cluster} test auc: {100*cluster_auc:.2f}')
    print(f'(perfect) {cluster} test acc: {100*p_cluster_acc:.2f} - {cluster} test auc: {100*p_cluster_auc:.2f}')
