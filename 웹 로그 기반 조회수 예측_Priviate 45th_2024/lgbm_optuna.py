import pandas as pd
import numpy as np
import random
import os
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
val_data = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_features, free_raw_data=False)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    clf = lgb.train(params, train_data, valid_sets=[val_data])

    y_pred = clf.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)**0.5

    return mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_trial.params

print(best_params)

# # Train the final model with the best hyperparameters
# final_model = lgb.train(best_params, train_data, valid_sets=[val_data], early_stopping_rounds=50, verbose_eval=False)
#
# # Make predictions on the test set
# pred = final_model.predict(df_test)
# pred = [0 if i < 0 else i for i in pred]
#
# df_submit = pd.read_csv('sample_submission.csv')
# df_submit['TARGET'] = pred
#
# df_submit.head()
#
# df_submit.to_csv("sample_submission.csv", index=False)
