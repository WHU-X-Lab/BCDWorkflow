import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#-----------------draw_heatmap------------------
# data = pd.read_excel('data.xlsx', engine='openpyxl')
# print(data.head())
# # 尝试将数据强制转换为数值类型
# data = data.apply(pd.to_numeric, errors='coerce')
# # 检查是否有空值（NaN），这些需要处理
# print(data.isnull().sum())
# plt.figure()
# heatmap = sns.heatmap(data=data, cmap='RdBu_r', vmin=0, vmax=1,annot=True,
#             fmt='.3f', linewidths=1.6,linecolor='white',annot_kws={"size": 12},
#             xticklabels=False, yticklabels=False,cbar_kws={'orientation': 'horizontal', 'shrink': 0.6,'pad': 0.06},)
# # 设置colorbar（图例）的字体大小和刻度大小
# colorbar = heatmap.collections[0].colorbar
# colorbar.ax.xaxis.set_tick_params(labelsize=12)  # 设置图例字体大小为 10
#
# plt.show()

#----------------draw_journal_data-----------------
import squarify

# 读取CSV文件
data = pd.read_csv('Journal_Distribution.csv')

# 数据准备
journal_counts = data['JournalName'].value_counts().reset_index()
journal_counts.columns = ['Journal', 'Count']
categories = data['JournalName']
values = data['Count']
# 计算蓝色渐变色
norm = plt.Normalize(min(values), max(values))  # 标准化数据
colors = plt.cm.Greens(norm(values))  # 使用蓝色渐变
# 绘制矩形分布图
plt.figure(figsize=(10, 6))
squarify.plot(sizes=values, label=categories, alpha=0.8, color=colors,linewidth=1,edgecolor='black')

plt.title("Distribution of Journal Categories", fontsize=14)
plt.axis('off')  # 隐藏坐标轴
plt.show()

# 读取 Zotero 导出的 CSV 文件
# data = pd.read_csv("Publication_Years2.csv")
#
# # 检查数据字段
# print(data.columns)
#
# # 统计发表年份分布
# years = data['PublicationYear']
# values = data['Count']
# # year_counts = data['PublicationYear'].value_counts().sort_index()
# norm = plt.Normalize(min(values), max(values))  # 标准化数据
# colors = plt.cm.Greens(norm(values))  # 使用蓝色渐变
# # 绘制饼图：文献发表年份分布
# plt.pie(
#     values,
#     labels=years,
#     autopct='%1.1f%%',
#     startangle=90,
#     colors=colors,
#     wedgeprops={'edgecolor':'white'},
# )
# plt.ylabel("")  # 隐藏 Y 轴标签
# plt.show()