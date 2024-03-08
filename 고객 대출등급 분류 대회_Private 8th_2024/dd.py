import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import re
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from itertools import combinations

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['대출기간'] = train['대출기간'].apply(lambda x: int(re.search(r'\d+', x).group()))
test['대출기간'] = test['대출기간'].apply(lambda x: int(re.search(r'\d+', x).group()))

train['월상환액_대출금액_비율'] = (train['총상환원금'] + train['총상환이자']) / (train['대출금액'] / (train['대출기간']))
test['월상환액_대출금액_비율'] = (test['총상환원금'] + test['총상환이자']) / (test['대출금액'] / (test['대출기간']))

train['상환_대비_대출금_비율'] = train['총상환원금'] / train['대출금액']
test['상환_대비_대출금_비율'] = test['총상환원금'] / test['대출금액']

train['대출_상환_비율'] = (train['총상환원금'] + train['총상환이자']) / train['대출금액']
test['대출_상환_비율'] = (test['총상환원금'] + test['총상환이자']) / test['대출금액']

train.drop(columns = ['연체계좌수', '최근_2년간_연체_횟수', '주택소유상태', '대출목적', '총계좌수', '근로기간', '총연체금액'], inplace=True)
test.drop(columns = ['연체계좌수', '최근_2년간_연체_횟수', '주택소유상태', '대출목적', '총계좌수', '근로기간', '총연체금액'], inplace=True)

train.set_index('ID', inplace=True)
test.set_index('ID', inplace=True)

def remove_outliers(data, column_name, grade_column='대출등급', alpha=0.00044):
  data_no_outliers = pd.DataFrame()
  for grade in data[grade_column].unique():
      subset = data[data[grade_column] == grade]
      lower_limit = subset[column_name].quantile(alpha / 2)
      upper_limit = subset[column_name].quantile(1 - alpha / 2)
      subset_no_outliers = subset[(subset[column_name] >= lower_limit) & (subset[column_name] <= upper_limit)]
      data_no_outliers = pd.concat([data_no_outliers, subset_no_outliers])
  return data_no_outliers
cols = train.select_dtypes(exclude=object).columns
for col in cols:
  train = remove_outliers(train, col, '대출등급')

cols = train.select_dtypes(exclude=object).columns
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

# tenure_mapping = {
#     '10+ years': 11,
#     '2 years': 3,
#     '< 1 year': 1,
#     '3 years': 4,
#     '1 year': 2,
#     'Unknown': 0,
#     '5 years': 6,
#     '4 years': 5,
#     '8 years': 9,
#     '6 years': 7,
#     '7 years': 8,
#     '9 years': 10,
#     '10+years': 11,
#     '<1 year': 1,
#     '3': 4,
#     '1 years': 2
# }
# train['근로기간'] = train['근로기간'].map(tenure_mapping)
# test['근로기간'] = test['근로기간'].map(tenure_mapping)

if '대출목적' in test.columns:
  test['대출목적'] = test['대출목적'].replace('결혼', '휴가')

# cols = test.select_dtypes(include=object).columns
# for col in cols:
#     le = LabelEncoder()
#     train[col] = le.fit_transform(train[col])
#     test[col] = le.transform(test[col])

map_list = {'A' : 6, 'B' : 5, 'C' : 4, 'D' : 3,  'E' : 2,  'F' : 1, 'G' : 0}
for key, item in map_list.items():
  train['대출등급'] = train['대출등급'].replace(key, item)

X = train.drop(columns=['대출등급'])
y = train['대출등급']

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import learning_curve

# 매크로 F1 점수를 계산하는 함수
def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


# Learning Curve를 그리는 함수 (매크로 F1 Score 기준)
def plot_learning_curve_f1(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Macro F1 Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=make_scorer(macro_f1_score))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 모델 생성
model = RandomForestClassifier(n_estimators=190,
                               max_depth=15,
                               min_samples_split=4,
                               max_features=None,
                               random_state=1111)
# Learning Curve 그리기 (매크로 F1 Score 기준)
plot_learning_curve_f1(model, "Learning Curve (Macro F1 Score)", X, y, cv=5, n_jobs=-1)

plt.show()
