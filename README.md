# sklearn_decision_tree #
自己最近有在學習一些機器學習的東西  
想要試著使用sklearn套件建立決策樹  
內容主要是演示如何建立出[字母辨識](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)這個資料集的決策樹
## 介紹決策樹 ##
決策樹是一種監督式機器學習模型  
適用於分類和回歸的預測  
主要的目的是建立出一個樹狀架構的模型讓資料依照決策樹的路徑來分類
## 開發環境及套件版本 ##
我使用NotePad++編寫python  
使用到的套件包含
* python 3.6.5
* numpy 1.15.3
* matplotlib 2.2.2
* scikit-learn 0.20.0
* pydotplus 2.0.2
* xlsxwriter 1.1.2
* graphviz 0.10.1 **[(安裝完後需要設定環境變數)](https://www.graphviz.org/download/)**
## 流程講解 ##
1. **尋找並下載資料集**  
   可以從 [UCI Machine Learning Repository: Data Sets](https://archive.ics.uci.edu/ml/datasets.html) 裡面挑選適合的  
   
2. **對資料集進行預處理**  
   將每一欄的數值進行標準化的動作  
   這個步驟的主要目的是讓欄位之間的資料落在相同的尺度範圍
   
3. **將資料集切割為訓練資料以及測試資料**  
