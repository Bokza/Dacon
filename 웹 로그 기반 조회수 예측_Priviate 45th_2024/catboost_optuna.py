import pandas as pd
import numpy as np
import random
import os
import optuna
from catboost import CatBoostRegressor, Pool
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
df_train = df_train.drop(['sessionID','userID'],axis=1)
df_test = df_test.drop(['sessionID','userID'],axis=1)

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

train_pool = Pool(data=x_train, label=y_train, cat_features=categorical_features)
val_pool = Pool(data=x_val, label=y_val, cat_features=categorical_features)

def objective(trial):
    # params = {
    #     'iterations': trial.suggest_int('iterations', 500, 2000),
    #     'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
    #     'depth': trial.suggest_int('depth', 5, 10),
    #     # Add more hyperparameters as needed
    # }
    params = {
        'iterations':trial.suggest_int("iterations", 500, 8000),
        'learning_rate' : trial.suggest_float('learning_rate',0.001, 0.15),
        'depth': trial.suggest_int('depth', 1, 15),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0.1, 30),
        'random_strength': trial.suggest_float('random_strength',1,50),
        # 'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00)
    }
    # params = {
    #     'iterations': trial.suggest_int('iterations', 500, 2000),
    #     'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
    #     'depth': trial.suggest_int('depth', 4, 10),
    #     'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-9, 10),
    #     'border_count': trial.suggest_int('border_count', 32, 255),
    #     'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.0, 10.0),
    #     'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10),
    # }
    clf = CatBoostRegressor(**params, random_state=42, verbose=0)
    clf.fit(train_pool, eval_set=[(val_pool)], early_stopping_rounds=50, verbose=0)
    # clf.fit(train_pool, eval_set=[(val_pool)], early_stopping_rounds=50, verbose=0)

    y_pred = clf.predict(val_pool)
    mse = mean_squared_error(y_val, y_pred)**0.5

    return mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_trial.params

print(best_params)

# # Train the final model with the best hyperparameters
# final_model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
# final_model.fit(train_pool, eval_set=[(val_pool)], early_stopping_rounds=50, verbose=0)
#
# # Make predictions on the test set
# test_pool = Pool(data=df_test, cat_features=categorical_features)
# pred = final_model.predict(test_pool)
# pred = [0 if i < 0 else i for i in pred]
#
# df_submit = pd.read_csv('sample_submission.csv')
# df_submit['TARGET'] = pred
#
# df_submit.head()
#
# df_submit.to_csv("sample_submission.csv", index=False)
