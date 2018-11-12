from sklearn import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from category_encoders import OrdinalEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

def do_cross_varify(criterion):
    decision_tree_classifier = DecisionTreeClassifier(criterion = criterion, max_depth = depth)
    
    score = cross_val_score(decision_tree_classifier, feature_attributes, class_attribute, cv = 10)
    
    return score.mean()
    
    
def draw_image(accuracy_of_gini, accuracy_of_entropy):
    plt.figure()

    plt.xticks(np.arange(2, 31, step = 2))

    plt.yticks(np.arange(0.7, 1, step = 0.05))
    
    plt.title('Decision Tree')
    
    plt.xlabel('the depth of decision tree')
    
    plt.ylabel('accuracy')

    plt.plot(depth_list, accuracy_of_gini, marker = 'o', label = 'gini')
    
    plt.plot(depth_list, accuracy_of_entropy, marker = 'o', label = 'entropy')
    
    plt.legend(loc = 'best')
    
    plt.savefig('./Desktop/sklearn_decision_tree-master/compare_tree.png')
    

if __name__ == '__main__':

    #讀取原始資料
    original_dataset = np.loadtxt('./Desktop/sklearn_decision_tree-master/car.data', dtype = 'str', delimiter  = ',')

    #隨機打散資料
    np.random.shuffle(original_dataset)

    #指定屬性
    feature_attributes = original_dataset[:, 0:6]

    #指定類別
    class_attribute = original_dataset[:, 6]

    #將屬性欄位轉換為數值
    feature_attributes = OrdinalEncoder().fit_transform(feature_attributes)
    feature_attributes = feature_attributes.values

    accuracy_of_gini = []

    accuracy_of_entropy = []

    depth_list = range(2, 30)

    #進行評估
    for depth in depth_list:
        accuracy_of_gini.append(do_cross_varify('gini'))

        accuracy_of_entropy.append(do_cross_varify('entropy'))
    
    #繪圖
    draw_image(accuracy_of_gini, accuracy_of_entropy)
