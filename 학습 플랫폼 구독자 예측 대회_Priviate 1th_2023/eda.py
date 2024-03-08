import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import boxplot

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
train_org = pd.read_csv('train.csv')
test_org = pd.read_csv('test.csv')
test_org['target'] = np.NaN
train_df = train_org.copy()
test_df = test_org.copy()

############################## Box Plot ##############################
cols = train_df.drop(columns='payment_pattern').select_dtypes(exclude='object').columns
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Malgun Gothic'
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
fig.suptitle("Box Plots for Various Features", fontsize=16)
titles = ["서비스 가입 기간","마지막으로 로그인한 시간","일반적인 로그인 시간",
                                      "학습 세션에 소요된 평균 시간","월간 학습 일수","완료한 총 코스 수",
                                      "최근 학습 성취도","중단된 학습 세션 수","커뮤니티 참여도",
                                      "고객 문의 이력", 'target']
for i, col in enumerate(cols):
    row_num = i // 3
    col_num = i % 3
    sns.boxplot(x='target', y=col, data=train_df, ax=axes[row_num, col_num], palette="Paired")
    axes[row_num, col_num].set_title(titles[i])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#######################################################################

############################## Pair Plot ##############################
sns.pairplot(train_df.rename(columns={'average_login_time':'일반적인 로그인 시간',
                         'average_time_per_learning_session':'학습 세션에 소요된 평균 시간',
                         'total_completed_courses':'완료한 총 코스 수',
                         'recent_learning_achievement':'최근 학습 성취도'})[['일반적인 로그인 시간', '학습 세션에 소요된 평균 시간', '완료한 총 코스 수', '최근 학습 성취도', 'target']], hue='target')
plt.show()
print(1)
#######################################################################

############################## Class count ##############################
df = pd.read_csv('submission_torchMLP_1204_best.csv')
plt.rcParams['font.family'] = 'Malgun Gothic'
sns.countplot(x='target', data=df, palette='Set2', hue='target')
plt.title('예측 Class 분포')
plt.show()

plt.rcParams['font.family'] = 'Malgun Gothic'
sns.countplot(x='target', data=train_org, palette='Set2', hue='target')
plt.title('학습 데이터 Class 분포')
plt.show()
#######################################################################