import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=[]
features=[]
labels=[]
csv_file = csv.reader(open('attachment1.csv'))
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content)
        features.append(content[0:8])
        labels.append(content[-1])
print('data=',data)
print('features=',features)
print('labels=',labels)
# scaler = StandardScaler() # 标准化转换
# scaler.fit(traffic_feature)  # 训练标准化对象
# traffic_feature= scaler.transform(traffic_feature)   # 转换数据集
feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.2,random_state=0)
regressor = RandomForestRegressor(n_estimators = 100,oob_score = True)
regressor.fit(feature_train,target_train)
predict_results=regressor.predict(feature_test)
print(explained_variance_score(predict_results, target_test))
print(mean_squared_error(target_test,predict_results))
print(regressor.score(feature_train,target_train))

# predict_visualization = regressor.predict(features)
# for i in range(0,len(predict_visualization)):
#     if len(predict_visualization)!=0:
#         if predict_visualization[i] <= 1.0 and predict_visualization[i] >= 0.75:
#             predict_visualization[i] = 1
#         elif predict_visualization[i] < 0.75 and predict_visualization[i] >= 0.50:
#             predict_visualization[i] = 0.66
#         elif predict_visualization[i] < 0.50 and predict_visualization[i] >= 0.25:
#             predict_visualization[i] = 0.33
#         else :
#             predict_visualization[i] = 0

# plt.title('fitting result')
# plt.plot(range(0,len(labels)), predict_visualization, color='red', label='training result')
# plt.plot(range(0,len(labels)), labels, color='green', label='original result')
# plt.legend()  # 显示图例
# plt.xlabel('number')
# plt.ylabel('labels')
# plt.show()
predict_visualization = regressor.predict(features)
loss = []
for i in  range(0,len(labels)):
    loss.append((labels[i] - predict_visualization[i])*(labels[i] - predict_visualization[i]))

plt.title('accuracy')
plt.plot(range(0,len(labels)),loss , color='red', label='training result')
plt.legend()  # 显示图例
plt.xlabel('number')
plt.ylabel('loss')
plt.show()


#可视化森林
from IPython.display import Image
from sklearn import tree
import pydotplus
import os
os.environ["Path"] += os.pathsep + 'D:\anaconda\anaconda3\pkgs\graphviz-2.38-hfd603c8_2\Library\bin'



Estimators = regressor.estimators_
for index, model in enumerate(Estimators):
    filename = 'tree_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=["input_sum","output_sum","input_num","output_num","input_sum_sd","output_sum_sd","input_num_sd","output_num_sd"],
                         class_names=None,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_pdf(filename)


#可视化权重
y_importances = regressor.feature_importances_
x_importances = ["input_sum","output_sum","input_num","output_num","input_sum_sd","output_sum_sd","input_num_sd","output_num_sd"]
y_pos = np.arange(len(x_importances))
# 横向柱状图
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.show()

# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0,1)
plt.title('Features Importances')
plt.show()


#文字树可视化
# for index,model in enumerate(Estimators):
#     r = tree.export_text(model,["input_sum","output_sum","input_num","output_num","input_sum_sd","output_sum_sd","input_num_sd","output_num_sd"])
#
print(regressor.feature_importances_)
attachment2_features = []
csv_file = csv.reader(open('attachment2.csv'))
#csv_file = csv.reader(open('attachment2_1.csv'))

for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        attachment2_features.append(content[0:8])

attachment2_results = regressor.predict(attachment2_features)
print(attachment2_results)

np.savetxt("new.csv", attachment2_results, delimiter=',')
print(regressor.max_samples)
