import pandas as pd
import numpy as np
from ta import *
import os

if not os.path.isdir('models'):
    os.mkdir('models')

asset = 'TEF'
investors = pd.read_csv(f'data/{asset}_Investors.csv')
prices = pd.read_csv(f'data/{asset}_Prices.csv')

# Use all TA features from 'ta' library.
ta_features_to_use = ['volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_vpt',
                      'volume_nvi', 'volatility_atr', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm',
                      'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                      'volatility_kchi', 'volatility_kcli', 'volatility_dch', 'volatility_dcl', 'volatility_dchi',
                      'volatility_dcli', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',
                      'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
                      'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_trix', 'trend_mass_index', 'trend_cci',
                      'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a',
                      'trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                      'trend_aroon_down', 'trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi',
                      'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao',
                      'others_dr', 'others_dlr', 'others_cr']
features_to_use = ['position', 'position_1m', 'position_3m', 'position_6m', 'position_1y'] + ta_features_to_use

ta_features = add_all_ta_features(prices, "open", "max", "min", "close", "volume", fillna=True)
ta_features = ta_features[['date'] + ta_features_to_use]

# Some of these financial indicators use close & volume, which are not available for prediction at date t.
ta_features[ta_features_to_use] = ta_features[ta_features_to_use].shift(fill_value=0.)

# We will only try to predict Buy moves here - we hypothesize that Sell moves, for 'retail' investors, may come from
# factors that do not depend on the market, whereas Buy moves are always motivated by market-related factors.
investors['previous_position'] = investors.groupby(['investor_id'])['position'].shift(fill_value=0.)
investors['move'] = investors.position - investors.previous_position
investors['buyer'] = 1 * (investors.move > 0.)

# As we use position in our model features, we need to shift it as well.
investors['position'] = investors['previous_position']

# We re-use the active definition introduced in the original paper, but on the Buy side only. We apply this threshold
# on the training set, as we need to see a client being active to be able to predict its activity.
investors_activity = investors[investors.date < '2006-01-01'].groupby('investor_id')['buyer'].sum()
active_investors = investors_activity[investors_activity > 20].index
print(f'Active investors: {len(active_investors)}')

# Creating a new investor encoding.
investors_ = active_investors
investor_mapping = dict(zip(active_investors, range(len(active_investors))))

data = investors.merge(ta_features, on=['date'], how='left')
data = data[data.investor_id.isin(active_investors)]
data['date'] = data['date'].apply(lambda x: float(x.replace('-', '')))
data['investor_encoding'] = data['investor_id'].apply(lambda x: investor_mapping[x])

# Removing irrelevant lines.
data = data[-(data.date == 20000103.)].reset_index()

# Classic setup - testing forward.
train_idx = (data.date < 20060101.)
val_idx = ((data.date >= 20060101.) & (data.date < 20070101.))
test_idx = (data.date >= 20070101.)

# We normalize position over clients, using means and stds computed on the training set only.
train_mean = data[train_idx].groupby('investor_encoding')['position'].mean()
train_mean = train_mean.reset_index().rename(columns={'position': 'mean'})
train_std = data[train_idx].groupby('investor_encoding')['position'].std()
train_std = train_std.reset_index().rename(columns={'position': 'std'})

data = data.merge(train_mean, on='investor_encoding', how='left')
data = data.merge(train_std, on='investor_encoding', how='left')

data['position'] = (data['position'] - data['mean']) / (data['std'] + 1e-7)
data['position_1m'] = data['position'].rolling(window=22, min_periods=1).mean()
data['position_3m'] = data['position'].rolling(window=66, min_periods=1).mean()
data['position_6m'] = data['position'].rolling(window=132, min_periods=1).mean()
data['position_1y'] = data['position'].rolling(window=264, min_periods=1).mean()
data = data[['date', 'investor_encoding', 'buyer'] + features_to_use]

# Saving splits as well.
data['train_idx'] = train_idx
data['val_idx'] = val_idx
data['test_idx'] = test_idx
data.to_csv(f'data/IBEX_{asset}_dataset.csv', index=False)
