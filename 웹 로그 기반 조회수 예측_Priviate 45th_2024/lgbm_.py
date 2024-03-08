import pandas as pd
import numpy as np
import random
import os
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed = 42
seed_everything(seed)  # Seed 고정

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = df_train.drop(['sessionID','userID'], axis=1)
df_test = df_test.drop(['sessionID','userID'], axis=1)

df_train.fillna('NAN', inplace=True)
df_test.fillna('NAN', inplace=True)

categorical_features = [
    "browser",
    "OS",
    "device",
    "continent",
    "subcontinent",
    "country",
    "traffic_source",
    "traffic_medium",
    "keyword",
    "referral_path",
]

for i in categorical_features:
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')

x_train = df_train.drop('TARGET', axis=1)
y_train = df_train['TARGET']

# Split the data into train and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
# val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_features, free_raw_data=False)


# score_l = []
# kfold = KFold(n_splits=5, shuffle=True, random_state=1111)
# for train_idx, val_idx in kfold.split(x_train, y_train):
#     x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
#     y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
#
#     train_data = lgb.Dataset(x_train_fold, label=y_train_fold, categorical_feature=categorical_features, free_raw_data=False)
#     params = {'num_leaves': 133, 'learning_rate': 0.07817856572815295, 'max_depth': 12,
#               'feature_fraction': 0.6945481659800496, 'bagging_fraction': 0.8411828542609351, 'bagging_freq': 1,
#               'min_child_samples': 74}
#     model = lgb.train(params, train_data)
#
#     pred = model.predict(x_val_fold)
#     pred = [0 if i < 0 else i for i in pred]
#
#     rmse = mean_squared_error(y_val_fold, pred)**0.5
#     score_l.append(rmse)
# print(score_l, sum(score_l)/5)

params = {'num_leaves': 133, 'learning_rate': 0.07817856572815295, 'max_depth': 12, 'feature_fraction': 0.6945481659800496, 'bagging_fraction': 0.8411828542609351, 'bagging_freq': 1, 'min_child_samples': 74}
model = lgb.train(params, train_data)

# Make predictions on the test set
pred = model.predict(df_test)
pred = [0 if i < 0 else i for i in pred]

df_submit = pd.read_csv('sample_submission.csv')
df_submit['TARGET'] = pred

df_submit.head()

df_submit.to_csv("sample_submission_lgb.csv", index=False)
