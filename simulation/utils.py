import numpy as np


def analyse_repartition(repartition, clients):

    found_mapping = np.argmax(repartition, axis=1)
    clients_group = np.array([client.split('_')[0] for client in clients])

    for dominant in set(found_mapping):
        current_cluster_idx = (found_mapping == dominant)
        current_groups = clients_group[current_cluster_idx]

        active_idx = current_cluster_idx & (clients_group == 'active')
        common_idx = current_cluster_idx & (clients_group == 'common')
        inactive_idx = current_cluster_idx & (clients_group == 'inactive')

        active_proba = 100 * np.mean(repartition[active_idx, dominant])
        active_proba = active_proba if not np.isnan(active_proba) else 0.
        common_proba = 100 * np.mean(repartition[common_idx, dominant])
        common_proba = common_proba if not np.isnan(common_proba) else 0.
        inactive_proba = 100 * np.mean(repartition[inactive_idx, dominant])
        inactive_proba = inactive_proba if not np.isnan(inactive_proba) else 0.

        print('Expert {0}:'.format(dominant))
        print('Overall mean proba : {0:.2f}'.format(100 * np.mean(repartition[:, dominant])))
        print('Active clients: {0} - mean proba: {1:.2f}'.format(np.sum(current_groups == 'active'), active_proba))
        print('Common clients: {0} - mean proba: {1:.2f}'.format(np.sum(current_groups == 'common'), common_proba))
        print('Inactive clients: {0} - mean proba: {1:.2f}\n'.format(np.sum(current_groups == 'inactive'), inactive_proba))
