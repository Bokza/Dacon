import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import optuna
import numpy as np
import warnings
warnings.filterwarnings('ignore')

var_list = ['최저기온', '일교차', '평균습도', '일사합', '강수량']
result_df = pd.DataFrame()
for i, var in enumerate(var_list):
    df = pd.read_csv('train.csv')
    df['강수량'] = df['강수량'].fillna(0)     # 강수량 널값은 비가 안와서 널값이라고 추측
    df = df.iloc[4749:]                     # 일사합 컬럼이 인덱스 4748까지 널값임
    df.interpolate(inplace=True)            # TimeSeriesData에는 선형 보간이 적합하다고 판단
    df.reset_index(drop=True, inplace=True)
    df['일시'] = pd.to_datetime(df['일시'])
    if var == '강수량':
        df[f'{var}'] = np.where(df['강수량'] < 3, 0,
                                np.where((df['강수량'] >= 3) & (df['강수량'] < 15), 1,
                                np.where((df['강수량'] >= 15) & (df['강수량'] < 30), 2,
                                         3)))
    df.rename(columns={'일시':'ds', f'{var}':'y'}, inplace=True)

    scaler = MinMaxScaler()
    df['y'] = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    #모델 학습
    # model = Prophet()
    model = Prophet(growth='linear',
                    weekly_seasonality = False,
                    changepoint_prior_scale = 0.1,
                    holidays_prior_scale = 0.01,
                    seasonality_prior_scale = 10,
                    seasonality_mode = 'multiplicative',
                    changepoint_range = 0.9
                    )
    model.add_country_holidays(country_name='KR')

    model.fit(df)

    #모델 예측
    future_data = model.make_future_dataframe(periods = 358, freq = 'd')
    forecast_data = model.predict(future_data)
    sub = forecast_data[['ds', 'yhat']].rename(columns={'ds':'일시', 'yhat':f'{var}'})[-358:]
    sub[f'{var}'] = scaler.inverse_transform(sub[f'{var}'].values.reshape(-1, 1))
    if i == 0:
        result_df = pd.concat([result_df, sub], axis=1)
    else:
        result_df = pd.concat([result_df, sub[f'{var}']], axis=1)

result_df.to_csv('프로펫_독립변수_20231229.csv', index=False, encoding='utf-8')

result_df = pd.read_csv('프로펫_독립변수_20231229.csv', encoding='utf-8')
result_df.set_index('일시', inplace=True)
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/서울 2023 기온 예측/train.csv')
df['강수량'] = df['강수량'].fillna(0)     # 강수량 널값은 비가 안와서 널값이라고 추측
df = df.iloc[4749:]                     # 일사합 컬럼이 인덱스 4748까지 널값임
df.interpolate(inplace=True)            # TimeSeriesData에는 선형 보간이 적합하다고 판단
df.reset_index(drop=True, inplace=True)
df['일시'] = pd.to_datetime(df['일시'])
df.set_index('일시', inplace=True)
df['강수량'] = np.where(df['강수량'] < 3, 0,
                       np.where((df['강수량'] >= 3) & (df['강수량'] < 15), 1,
                       np.where((df['강수량'] >= 15) & (df['강수량'] < 30), 2,
                       3)))

df = df[var_list+['평균기온']]

scaler = MinMaxScaler()
cols = df.drop(columns=['평균기온']).columns
df[cols] = scaler.fit_transform(df[cols])
result_df[cols] = scaler.transform(result_df[cols])

# best model -> filename : Prophet_XGBRegressor_Optuna_231229.csv
model = XGBRegressor(n_estimators = 150, 
                     colsample_bytree=0.650938788194843, 
                     learning_rate=0.02997152666801559, 
                     subsample=0.8654298963125641) # best
model.fit(df.drop(columns=['평균기온']), df['평균기온'])
pred =  model.predict(result_df)
sub = pd.DataFrame(pd.date_range(start='2023-01-01', end='2023-12-24', freq='D'), columns=['일시'])
sub['평균기온'] = pred
sub.to_csv('Prophet_XGBRegressor_Optuna_231229.csv', index=False, encoding='utf-8')

########################################################################################################################
# 하이퍼 파라미터 튜닝 Optuna
X = df.drop(columns=['평균기온'])
y = df['평균기온']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 150),
        'subsample': trial.suggest_uniform('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 0.7),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.03),
    }

    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)

    return mae

# 최적화
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 출력
print(study.best_params)