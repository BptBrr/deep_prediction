import sys
sys.path.append('..')
from exnet_v3 import *
from utils import *
import pandas as pd
import numpy as np
import pickle
import umap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed = 0
np.random.seed(seed)

# Build params
n_experts = 16
expert_architecture = [64, 32]
embedding_size = 64
dropout_rates = {'input': 0., 'hidden': 0.2}
weight_decay = {'l1': 0., 'l2': 0.}
spec_weight = 0.01
entropy_weight = 0.1
gamma = 0.

# Fit params
n_epochs = 200
patience = 20
batch_size = 128
learning_rate = 1e-3
optimizer = 'nadam'
lookahead = True

data = pd.read_csv('data/sampled_data.csv')
with open('data/sampled_data_specs.pkl', 'rb') as f:
    data_specs = pickle.load(f)

features = [feature for feature in data.columns if 'X' in feature]
clients = data.Client.unique()
clients_groups = [client.split('_')[0] for client in clients]

np.random.shuffle(clients)  # Clients are shuffled so as not to provide info to the model through clients' encodings.
clients_encoding = dict(zip(clients, range(len(clients))))
encoding_to_group = {clients_encoding[c]: c.split('_')[0] for c in clients}
data['client_encoding'] = [clients_encoding[client] for client in data.Client]

# Splitting & shuffling data.
train_data = data.iloc[data_specs['train_split']].sample(frac=1.)
val_data = data.iloc[data_specs['valid_split']].sample(frac=1.)
test_data = data.iloc[data_specs['test_split']].sample(frac=1.)

train_data_ = (train_data[features].values.astype(np.float32), train_data.client_encoding.values.astype(np.int32),
               pd.get_dummies(train_data.Y).values.astype(np.float32))
val_data_ = (val_data[features].values.astype(np.float32), val_data.client_encoding.values.astype(np.int32),
             pd.get_dummies(val_data.Y).values.astype(np.float32))
test_data_ = (test_data[features].values.astype(np.float32), test_data.client_encoding.values.astype(np.int32),
              pd.get_dummies(test_data.Y).values.astype(np.float32))

print(f'Training on {train_data_[0].shape[0]} samples, validating on {val_data_[0].shape[0]} samples.')
exnet_model = ExNet(n_feats=5, output_dim=2, n_experts=n_experts, expert_architecture=expert_architecture,
                    n_investors=len(clients), embedding_size=embedding_size, dropout_rates=dropout_rates,
                    weight_decay=weight_decay,  spec_weight=spec_weight, entropy_weight=entropy_weight, gamma=gamma,
                    name=f'ExNet_{n_experts}_s{spec_weight}_e{entropy_weight}')
exnet_model.fit(train_data=train_data_, val_data=val_data_, n_epochs=n_epochs, batch_size=batch_size,
                optimizer=optimizer, learning_rate=learning_rate, patience=patience, lookahead=lookahead,
                save_path='models/', seed=seed)

train_pred = exnet_model.predict(train_data_[0:2])
val_pred   = exnet_model.predict(val_data_[0:2])
test_pred  = exnet_model.predict(test_data_[0:2])

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
    cluster_pred = exnet_model.predict((test_data_[0][cluster_idx], test_data_[1][cluster_idx]))[:, 1]
    cluster_acc = accuracy_score(test_data.iloc[cluster_idx].Y.values, cluster_pred > .5)
    cluster_auc = roc_auc_score(test_data.iloc[cluster_idx].Y.values, cluster_pred)
    p_cluster_acc = accuracy_score(test_data.iloc[cluster_idx].Y.values, test_data.iloc[cluster_idx].proba.values > .5)
    p_cluster_auc = roc_auc_score(test_data.iloc[cluster_idx].Y.values, test_data.iloc[cluster_idx].proba.values)
    print(f'{cluster} test acc: {100*cluster_acc:.2f} - {cluster} test auc: {100*cluster_auc:.2f}')
    print(f'(perfect) {cluster} test acc: {100*p_cluster_acc:.2f} - {cluster} test auc: {100*p_cluster_auc:.2f}')

# ===== Clustering results =====
cmap = plt.get_cmap('jet')
idx = dict(zip(np.unique(clients_groups), np.linspace(0., 1., len(np.unique(clients_groups)))))

probas, _ = exnet_model.get_experts_repartition(print_stats=True)
exnet_model.plot_experts_repartition()
analyse_repartition(probas, clients)

projector = umap.UMAP(n_neighbors=5, min_dist=0.1)
proj_probas = projector.fit_transform(probas)
plt.scatter(proj_probas[:, 0], proj_probas[:, 1], color=[cmap(idx[encoding_to_group[i]]) for i in range(len(clients))])
plt.title('UMAP Visualization of Investors Embeddings')
plt.show()
