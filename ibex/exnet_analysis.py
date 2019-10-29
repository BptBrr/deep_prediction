from exnet_v3 import *
from sklearn.metrics import roc_auc_score, average_precision_score

asset = 'TEF'
build_params = {'n_experts': 4,
                'spec_weight': 7.7e-4,
                'entropy_weight': 4.2e-2,
                'expert_architecture': [32, 32],
                'embedding_size': 32,
                'dropout_rates': {'input': 0.1, 'hidden': 0.5},
                'weight_decay': {'l1': 0., 'l2': 0.},
                'gamma': 2.5}

data = pd.read_csv(f'data/IBEX_{asset}_dataset.csv')
features = list(data.columns[3:-3])  # Removing irrelevant columns - date, encoding, target & splits.

train_idx = (data.date < 20060101.)
val_idx = ((data.date >= 20060101.) & (data.date < 20070101.))
test_idx = (data.date >= 20070101.)

train_data = data[train_idx]
val_data = data[val_idx]
test_data = data[test_idx]

train_data_ = (train_data[features].values.astype(np.float32),
               train_data.investor_encoding.values.astype(np.int32),
               pd.get_dummies(train_data.buyer).values.astype(np.float32))

val_data_ = (val_data[features].values.astype(np.float32),
             val_data.investor_encoding.values.astype(np.int32),
             pd.get_dummies(val_data.buyer).values.astype(np.float32))

test_data_ = (test_data[features].values.astype(np.float32),
              test_data.investor_encoding.values.astype(np.int32),
              pd.get_dummies(test_data.buyer).values.astype(np.float32))

full_data = (data[features].values.astype(np.float32),
             data.investor_encoding.values.astype(np.int32),
             pd.get_dummies(data.buyer).values.astype(np.float32))

exnet_model = ExNet(n_feats=63, output_dim=2, n_investors=len(np.unique(train_data_[1])), name=f'ExNet_{asset}',
                    **build_params)
exnet_model.fake_call()
exnet_model.load_weights(f'ExNet_{asset}.h5')

probas, _ = exnet_model.get_experts_repartition(print_stats=True)
_, exp, _ = exnet_model(*full_data[0:2])
print(f'Global spec loss: {exnet_model.specialization_loss(exp):.5f}')

_ = exnet_model.plot_experts_repartition()
exnet_model.plot_experts_umap(n_neighbors=20)

train_pred = exnet_model.predict(train_data_[0:2])
val_pred = exnet_model.predict(val_data_[0:2])
test_pred = exnet_model.predict(test_data_[0:2])
full_pred = exnet_model.predict(full_data[0:2])

train_auc = roc_auc_score(train_data.buyer.values, train_pred[:, 1])
val_auc = roc_auc_score(val_data.buyer.values, val_pred[:, 1])
test_auc = roc_auc_score(test_data.buyer.values, test_pred[:, 1])

train_ap = average_precision_score(train_data.buyer.values, train_pred[:, 1])
val_ap = average_precision_score(val_data.buyer.values, val_pred[:, 1])
test_ap = average_precision_score(test_data.buyer.values, test_pred[:, 1])

print('----- Results -----')
print(f'train auc: {100*train_auc:.2f}\nval auc: {100*val_auc:.2f}\ntest auc: {100*test_auc:.2f}\n')
print(f'train ap: {100*train_ap:.2f}\nval ap: {100*val_ap:.2f}\ntest ap: {100*test_ap:.2f}\n')

print('----- Permutation Importance -----')
position_feats = [f for f in features if f.startswith('position')]
volume_feats = [f for f in features if f.startswith('volume')]
volatility_feats = [f for f in features if f.startswith('volatility')]
trend_feats = [f for f in features if f.startswith('trend')]
momentum_feats = [f for f in features if f.startswith('momentum')]
others_feats = [f for f in features if f.startswith('others')]

groups = [position_feats, volume_feats, volatility_feats, trend_feats, momentum_feats, others_feats]
repeats = 100

perm_importance = pd.DataFrame(columns=['cl', 'feat', 'evol'])
for encod in [205, 214, 293]:

    encod_idx = full_data[1] == encod
    ground_perf = average_precision_score(full_data[2][encod_idx][:, 1], full_pred[encod_idx][:, 1])

    for group in groups:
        perf_evol = 0.
        for i in range(repeats):
            pred_data = full_data[0][encod_idx].copy()
            for feat in group:
                np.random.shuffle(pred_data[:, features.index(feat)])

            new_perf = average_precision_score(full_data[2][encod_idx][:, 1],
                                               exnet_model.predict((pred_data, full_data[1][encod_idx]))[:, 1])
            perf_evol += (new_perf - ground_perf) / ground_perf
        perf_evol /= repeats

        perm_importance.loc[len(perm_importance)] = [encod, group[0].split('_')[0], perf_evol]

print('First group:')
print(perm_importance[perm_importance.cl == 205].sort_values(by='evol'))
print('\nSecond group:')
print(perm_importance[perm_importance.cl == 214].sort_values(by='evol'))
print('\nThird group:')
print(perm_importance[perm_importance.cl == 293].sort_values(by='evol'))
