# 2.6435767004870385
# 2.7373253887768514
# 2.415173508552896
# 2.688129093841945
# 2.967354363604868
# [2.6435767004870385, 2.7373253887768514, 2.415173508552896, 2.688129093841945, 2.967354363604868] 2.6903118110527195
# best

import pandas as pd
import numpy as np
import random
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


random_strength_list = [1.001, 1.0011, 1.0012, 1.0013, 1.0014, 1.0015, 1.0016, 1.0017, 1.0018, 1.0019,
                        1.002, 1.0021, 1.0022, 1.0023, 1.0024, 1.0025, 1.0026, 1.0027, 1.0028, 1.0029,
                        1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.01]

for item in random_strength_list:
    print(f'>>>>>>>>>>>>>>>>>>>> {item} 시작 >>>>>>>>>>>>>>>>>>>>')
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

    # df_train.drop(columns=[item], inplace=True)
    # df_test.drop(columns=[item], inplace=True)

    df_train.fillna('NAN', inplace=True)
    df_test.fillna('NAN', inplace=True)

    categorical_features = df_train.select_dtypes(include='object').columns.to_list()
    for i in categorical_features:
        df_train[i] = df_train[i].astype('category')
        df_test[i] = df_test[i].astype('category')

    x_train = df_train.drop('TARGET', axis=1)
    y_train = df_train['TARGET']

    score = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1111)
    for train_idx, val_idx in kfold.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_pool = Pool(data=x_train_fold, label=y_train_fold, cat_features=categorical_features)
        params = {'iterations': 799, 'learning_rate': 0.08634765067330821, 'depth': 9, 'random_strength':item}
        model = CatBoostRegressor(**params, random_state=seed, verbose=False)
        model.fit(train_pool)

        test_pool = Pool(data=x_val_fold, cat_features=categorical_features)
        pred = model.predict(x_val_fold)
        pred = [0 if i < 0 else i for i in pred]

        rmse = mean_squared_error(y_val_fold, pred)**0.5
        score.append(rmse)
        print(f'RMSE : {rmse}')
    print(score, sum(score)/5)

    # train_pool = Pool(data=x_train, label=y_train, cat_features=categorical_features)
    # params = {'iterations': 799, 'learning_rate': 0.08634765067330821, 'depth': 9}
    # clf = CatBoostRegressor(**params, random_state=seed)
    # clf.fit(train_pool)
    #
    # test_pool = Pool(data=df_test, cat_features=categorical_features)
    # pred = clf.predict(test_pool)
    #
    # pred = [0 if i < 0 else i for i in pred]
    #
    # df_submit = pd.read_csv('sample_submission.csv')
    # df_submit['TARGET'] = pred
    #
    # df_submit.head()
    #
    # df_submit.to_csv("cat_devicebounced_hyper.csv", index=False)
    #
    #
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