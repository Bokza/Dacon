import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
train.sort_values('대출등급', inplace=True)
train.pop('ID')
train.set_index('대출등급', inplace=True)


# 대출 등급에 따른 서브플롯 그리기
loan_grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), sharex=True, sharey=True)
for i, grade in enumerate(loan_grades):
    row = i // 4
    col = i % 4
    # sns.histplot(train.loc[grade, '대출금액'], bins=20, ax=axes[row, col], kde=True)
    sns.histplot(train.loc[grade, '대출금액'], bins=20, ax=axes[row, col], kde=True, stat="density", common_norm=False)
    axes[row, col].set_title(f'Loan Grade {grade}')
plt.tight_layout()
plt.show()

# 대출등급에 따른 주택소유상태의 비율을 계산
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

loan_grade_ratio = train.groupby(['대출등급', '주택소유상태']).size() / len(train) * 100
loan_grade_ratio = loan_grade_ratio.reset_index(name='비율')

# 등급별 서브 플롯
grades = sorted(train['대출등급'].unique())
fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharey=True)
fig.suptitle('주택소유상태에 따른 등급별 대출 비율')

palette = sns.color_palette("Blues", 7)

for i, grade in enumerate(grades):
    ax = axes[i // 4, i % 4]
    data = loan_grade_ratio[loan_grade_ratio['대출등급'] == grade]
    sns.barplot(x='주택소유상태', y='비율', data=data, ax=ax, palette=palette)
    ax.set_title(f'등급 {grade}')
    ax.set_xlabel('주택소유상태')
    ax.set_ylabel('대출 비율 (%)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print(1)