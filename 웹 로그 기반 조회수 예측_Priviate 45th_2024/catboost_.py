# KFold  2.718 정도였던 거 같음
# Public 2.926
# Early Stopping, 파생변수, 이상치처리, 하이퍼파라미터 추가 하면 될듯

import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
def calculate_vif(data_frame):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data_frame.columns
    vif_data["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]
    return vif_data


items = ['transaction_revenue', 'transaction', 'traffic_source']
# items = ['browser', 'subcontinent', 'OS']
# for item in items:
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed = 777
seed_everything(seed)  # Seed 고정

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# print(calculate_vif(df_train.select_dtypes(exclude='object').drop(columns=['TARGET']))) # VIF계수 계산
# sns.pairplot(df_train.select_dtypes(exclude='object').drop(columns=['TARGET']))
# plt.show()
df_train = df_train.drop(['sessionID','userID'],axis=1)
df_test = df_test.drop(['sessionID','userID'],axis=1)

###################################################파생변수 생성#####################################################
# 수치형 변수
df_train['weighted_quality'] = df_train['quality'] * (df_train['duration'])
df_test['weighted_quality'] = df_test['quality'] * (df_test['duration'])

ob_list = ['country', 'OS', 'browser', 'device', 'continent', 'subcontinent']
for ob in ob_list:
    df_train[f'mean_quality_by_{ob}'] = df_train.groupby(ob)['quality'].transform('mean')
    df_train[f'mean_duration_by_{ob}'] = df_train.groupby(ob)['duration'].transform('mean')
    df_test[f'mean_quality_by_{ob}'] = df_test.groupby(ob)['quality'].transform('mean')
    df_test[f'mean_duration_by_{ob}'] = df_test.groupby(ob)['duration'].transform('mean')
    df_train[f'mean_bounced_by_{ob}'] = df_train.groupby(ob)['bounced'].transform('mean')
    df_test[f'mean_bounced_by_{ob}'] = df_test.groupby(ob)['bounced'].transform('mean')
###################################################################################################################

df_train.fillna('NAN', inplace=True)
df_test.fillna('NAN', inplace=True)

categorical_features = df_train.select_dtypes(include='object').columns.to_list()
for i in categorical_features:
    df_train[i] = df_train[i].astype('category')
    df_test[i] = df_test[i].astype('category')

x_train = df_train.drop('TARGET', axis=1)
y_train = df_train['TARGET']

train_pool = Pool(data=x_train, label=y_train, cat_features=categorical_features)
params = {'iterations': 799, 'learning_rate': 0.08634765067330821, 'depth': 9, 'random_strength':1.005}
clf = CatBoostRegressor(**params, random_state=seed)
clf.fit(train_pool)

test_pool = Pool(data=df_test, cat_features=categorical_features)
pred = clf.predict(test_pool)

pred = [0 if i < 0 else i for i in pred]

df_submit = pd.read_csv('sample_submission.csv')
df_submit['TARGET'] = pred

df_submit.head()

df_submit.to_csv(f"cat2.csv", index=False)


# '''
# quality              0.450602
# userID               0.400613
# duration             0.089978
# new                  0.012182
# country              0.007863
# OS                   0.005711
# subcontinent         0.005663
# browser              0.005637
# traffic_source       0.004151
# transaction          0.003845
# transaction_revenue  0.003836
# referral_path        0.003815
# keyword              0.001803
# traffic_medium       0.001572
# continent            0.001510
# device               0.001198
# bounced              0.000020
# '''