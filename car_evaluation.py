from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pydotplus
import xlsxwriter
from category_encoders import OrdinalEncoder

#讀取原始資料
original_dataset = np.loadtxt('./Desktop/sklearn_decision_tree-master/car.data', dtype = 'str', delimiter  = ',')

#指定屬性
feature_attributes = original_dataset[:, 0:6]

#指定類別
class_attribute = original_dataset[:, 6]

#將屬性欄位轉換為數值
feature_attributes = OrdinalEncoder().fit_transform(feature_attributes)
feature_attributes = feature_attributes.values

#切割訓練及測試資料
feature_attributes_of_training_data, feature_attributes_of_test_data, class_attribute_of_training_data, class_attribute_of_test_data = train_test_split(feature_attributes, class_attribute, test_size = 0.3)

decision_tree_classifier = DecisionTreeClassifier()

#訓練決策樹
decision_tree = decision_tree_classifier.fit(feature_attributes_of_training_data, class_attribute_of_training_data)

prediction_of_training_data = decision_tree.predict(feature_attributes_of_training_data)
accuracy_of_training_data = metrics.accuracy_score(class_attribute_of_training_data, prediction_of_training_data)
print(accuracy_of_training_data)

prediction_of_test_data = decision_tree.predict(feature_attributes_of_test_data)
accuracy_of_test_data = metrics.accuracy_score(class_attribute_of_test_data, prediction_of_test_data)
print(accuracy_of_test_data)

#繪製決策樹
feature_name = np.array([['buying'], ['maint'], ['doors'], ['persons'], ['lug_boot'], ['safety']])
class_name = np.array(['acc', 'good', 'unacc', 'vgood'])

dot_data = tree.export_graphviz(decision_tree, feature_names = feature_name, class_names = class_name, filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)

try:
    graph.write_pdf('./Desktop/sklearn_decision_tree-master/decision_tree.pdf')
except:
    print('無法繪製新的決策樹\n請先關閉已經開啟的檔案\n並重新執行程式')

#新增excel檔案
workbook = xlsxwriter.Workbook('./Desktop/sklearn_decision_tree-master/predict.xlsx')

worksheet1 = workbook.add_worksheet('Car Evaluation')

title = np.array([['原始類別', 'buying\n\n1 = vhigh\n\n2 = high\n\n3 = med\n\n4 = low', 'maint\n\n1 = vhigh\n\n2 = high\n\n3 = med\n\n4 = low', 'doors\n\n2 = 1\n\n3 = 2\n\n4 = 3\n\n5more = 4', 'persons\n\n2 = 1\n\n4 = 2\n\nmore = 3', 'lug_boot\n\nsmall = 1\n\nmed = 2\n\nbig = 3', 'safety\n\nlow = 1\n\nmed = 2\n\nhigh = 3', '', '預測類別']])
column_1 = class_attribute_of_test_data.reshape((-1, 1))
column_3 = np.full((column_1.shape[0], 1), '→')
column_4 = prediction_of_test_data.reshape((-1, 1))

result = np.concatenate((column_1, feature_attributes_of_test_data, column_3, column_4), axis = 1)
result = np.concatenate((title, result), axis = 0)

#寫檔
for row, data in enumerate(result):
    worksheet1.write_row(row, 0, data)

try:
    workbook.close()
except:
    print('無法輸出新的預測結果\n請先關閉已經開啟的檔案\n並重新執行程式')
