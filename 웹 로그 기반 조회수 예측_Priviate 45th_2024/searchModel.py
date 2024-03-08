import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed = 1111
seed_everything(seed)  # Seed 고정

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = df_train.drop(['sessionID','userID'],axis=1)
df_test = df_test.drop(['sessionID','userID'],axis=1)
###################################################파생변수 생성#####################################################
# 수치형 변수
df_train['transaction_rate'] = (df_train['transaction'] / df_train['duration']).fillna(0)
df_train['weighted_quality'] = df_train['quality'] * (df_train['duration'])
df_test['transaction_rate'] = (df_test['transaction'] / df_test['duration']).fillna(0)
df_test['weighted_quality'] = df_test['quality'] * (df_test['duration'])

ob_list = ['country', 'OS', 'browser', 'device', 'continent', 'subcontinent']
for ob in ob_list:
    df_train[f'mean_quality_by_{ob}'] = df_train.groupby(ob)['quality'].transform('mean')
    df_train[f'mean_duration_by_{ob}'] = df_train.groupby(ob)['duration'].transform('mean')
    df_test[f'mean_quality_by_{ob}'] = df_test.groupby(ob)['quality'].transform('mean')
    df_test[f'mean_duration_by_{ob}'] = df_test.groupby(ob)['duration'].transform('mean')
###################################################################################################################

df_train.fillna('NAN', inplace=True)
df_test.fillna('NAN', inplace=True)

categorical_features = df_train.select_dtypes(include='object').columns.to_list()
for i in categorical_features:
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')

x_train = df_train.drop('TARGET', axis=1)
y_train = df_train['TARGET']

# score = []
# kfold = KFold(n_splits=5, shuffle=True, random_state=1111)
# for train_idx, val_idx in kfold.split(x_train, y_train):
#     x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
#     y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
#
#     params = {'iterations': 799, 'learning_rate': 0.08634765067330821, 'depth': 9}
#     lgbm_model = LGBMRegressor(random_state=seed, categorical_features=categorical_features)
#     catboost_model = CatBoostRegressor(**params, random_state=seed, cat_features=categorical_features)
#     meta_model = CatBoostRegressor(**params, random_state=seed)
#     model = StackingRegressor(
#         estimators=[('lgbm', lgbm_model), ('catboost', catboost_model)],
#         final_estimator=meta_model
#     )
#     model.fit(x_train_fold, y_train_fold)
#
#     pred = model.predict(x_val_fold)
#     pred = [0 if i < 0 else i for i in pred]
#
#     rmse = mean_squared_error(y_val_fold, pred)**0.5
#     score.append(rmse)
#     print(f'RMSE : {rmse}')
# print(score, sum(score)/5)

params = {'iterations': 5, 'learning_rate': 0.08634765067330821, 'depth': 9}
lgbm_model = LGBMRegressor(random_state=seed, categorical_features=categorical_features)
catboost_model = CatBoostRegressor(**params, random_state=seed, cat_features=categorical_features)
meta_model = CatBoostRegressor(**params, random_state=seed)
model = StackingRegressor(
    estimators=[('lgbm', lgbm_model), ('catboost', catboost_model)],
    final_estimator=meta_model
)
model.fit(x_train, y_train)

pred = model.predict(df_test, categorical_feature=categorical_features)
pred = [0 if i < 0 else i for i in pred]

df_submit = pd.read_csv('sample_submission.csv')
df_submit['TARGET'] = pred

df_submit.head()

df_submit.to_csv(f"stacking.csv", index=False)
