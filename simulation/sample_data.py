import numpy as np
import pandas as pd
import pickle
import os

# Creating required folders.
if not os.path.isdir('models'):
    os.mkdir('models')

if not os.path.isdir('data'):
    os.mkdir('data')

np.random.seed(1)

# Sampling parameters
n_samples = 100000
n_clients = 500
noise_level = 0.5
n_features = 5

valid_split = 0.2
test_split = 0.1


def sigmoid(x): return 1 / (1 + np.exp(-x))


def sample_groups(n_samples, n_clients):

    groups = ['active', 'inactive', 'common']

    attributed_samples = 0
    attributed_clients = 0
    groups_ = {}
    clients_ = []

    for group in groups:

        if group == 'active':
            # Active groups count for ~70% of trades, while covering ~10% of clients.
            samples_prct = 0.01 * np.random.standard_normal(1)[0] + 0.70
            clients_prct = 0.01 * np.random.standard_normal(1)[0] + 0.10
            samples = int(samples_prct * n_samples)
            clients = int(clients_prct * n_clients)

        elif group == 'inactive':
            # Inactive groups count for ~50% of clients, while covering ~10% of trades.
            samples_prct = 0.01 * np.random.standard_normal(1)[0] + 0.10
            clients_prct = 0.01 * np.random.standard_normal(1)[0] + 0.50
            samples = int(samples_prct * n_samples)
            clients = int(clients_prct * n_clients)

        else:
            # Common group takes the remaining clients & samples.
            samples = n_samples - attributed_samples
            clients = n_clients - attributed_clients

        # Equirepartition of samples between clients.
        samples_per_client = samples // clients
        counter = 0

        for client in range(clients - 1):
            clients_ += [group + '_client_{0}'.format(client + 1)] * samples_per_client
            counter += samples_per_client

        clients_ += [group + '_client_{0}'.format(clients + 1)] * (samples - counter)
        groups_[group] = {'n_samples': samples, 'n_clients': clients}

        attributed_samples += samples
        attributed_clients += clients

    return groups_, clients_


groups, clients = sample_groups(n_samples, n_clients)
clients_noise_ = np.random.normal(loc=0., scale=noise_level, size=(n_clients, n_features))
clients_noise = dict(zip(np.unique(clients), clients_noise_))

features = np.random.normal(size=(n_samples, n_features))
data = pd.DataFrame(features, columns=['X{0}'.format(i) for i in range(1, n_features + 1)])
data['Client'] = clients

# ExNet decision rules :
active_rule = [5, 5, 0, 0, -5]
inactive_rule = [-5, 0, 5, 0, 5]
common_rule = [-5, 0, 5, 5, 0]

features = data[['X{0}'.format(i) for i in range(1, n_features + 1)]]
clients_noise_ = np.array([clients_noise[cl] for cl in data.Client])

data['proba_active'] = sigmoid(np.sum(features * (active_rule + clients_noise_), axis=1))
data['proba_inactive'] = sigmoid(np.sum(features * (inactive_rule + clients_noise_), axis=1))
data['proba_common'] = sigmoid(np.sum(features * (common_rule + clients_noise_), axis=1))

data['proba'] = data['proba_active'] * [client.startswith('active') for client in clients] + \
                data['proba_inactive'] * [client.startswith('inactive') for client in clients] + \
                data['proba_common'] * [client.startswith('common') for client in clients]

data['Y'] = 1 * (data['proba'] > np.random.uniform(size=n_samples))

valid_set = np.random.choice(range(len(data)), size=int(valid_split*len(data)), replace=False)
test_set = np.random.choice(list(set(range(len(data))) - set(valid_set)), size=int(test_split*len(data)), replace=False)
train_set = list(set(range(len(data))) - set(valid_set) - set(test_set))

groups['active']['decision_rule'] = list(active_rule)
groups['inactive']['decision_rule'] = list(inactive_rule)
groups['common']['decision_rule'] = list(common_rule)
groups['train_split'] = list(train_set)
groups['valid_split'] = list(valid_set)
groups['test_split'] = list(test_set)
groups['clients_noise'] = clients_noise

data[['Client'] + ['X{0}'.format(i+1) for i in range(n_features)] + ['proba', 'Y']].to_csv('data/sampled_data.csv', index=False)
with open('data/sampled_data_specs.pkl', 'wb') as f:
    pickle.dump(groups, f)
