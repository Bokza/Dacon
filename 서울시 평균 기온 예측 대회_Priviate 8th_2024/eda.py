import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# data = pd.read_csv('./train.csv')
# data['년'] = data['일시'].str.slice(0, 4)
# data['월'] = data['일시'].str.slice(5, 7)
# data['월일'] = data['일시'].str.slice(5)
#
# # 연도/월별 온도
# agg = data.pivot_table('평균기온', '월', '년', 'median')
# fig = px.imshow(agg, x=agg.columns, y=agg.index, color_continuous_scale='Viridis')
# fig.update_layout(title='연도/월별 평균기온')
# fig.show()

df = pd.read_csv('./train.csv', parse_dates=['일시'])
sub = pd.read_csv('Prophet_XGBRegressor_Optuna_231229.csv')
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='GnBu')
# plt.tight_layout()
# plt.show()
#
# sns.pairplot(df[['최고기온', '최저기온', '평균습도', '일사합', '평균기온']])
# plt.show()


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
merged_df = pd.concat([df, sub])
merged_df['일시'] = pd.to_datetime(merged_df['일시'])
plt.figure(figsize=(18, 6))
sns.set_palette('Set2')
sns.lineplot(x='일시', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2020], label='2020년', linewidth=2.5)
sns.lineplot(x='일시', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2021], label='2021년', linewidth=2.5)
sns.lineplot(x='일시', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2022], label='2022년', linewidth=2.5)
sns.lineplot(x='일시', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2023], label='2023년(pred)', linewidth=2.5)
plt.xlabel('날짜')
plt.ylabel('평균기온')
plt.title('년도별 평균기온', pad=20, size=15)
plt.legend()
plt.tight_layout()
plt.show()


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
merged_df = pd.concat([df, sub])
merged_df['일시'] = pd.to_datetime(merged_df['일시'])
plt.figure(figsize=(18, 6))
sns.set_palette('Set2')
sns.lineplot(x='level_0', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2020].reset_index().reset_index(), label='2020년', alpha=0.5, linewidth=2.5)
sns.lineplot(x='level_0', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2021].reset_index().reset_index(), label='2021년', alpha=0.5, linewidth=2.5)
sns.lineplot(x='level_0', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2022].reset_index().reset_index(), label='2022년', alpha=0.5, linewidth=2.5)
sns.lineplot(x='level_0', y='평균기온', data=merged_df[merged_df['일시'].dt.year==2023].reset_index().reset_index(), label='2023년(pred)', alpha=1, linewidth=2.5)

plt.xlabel('날짜')
plt.ylabel('평균기온')
plt.title('년도별 평균기온', pad=20, size=15)
plt.legend()
plt.tight_layout()
plt.show()