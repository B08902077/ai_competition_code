# 模型訓練
本資料夾內含四個程式，分別使用不同模型進行訓練，並將預測結果寫入檔案，提供給其他程式(例如根目錄的`main.py`)使用  
- model_CB.py: 使用CatBoost  
- model.LGBM.py: 使用LightGBM  
- model.NN.py: 使用Neural Network  
- model.RF.py: 使用Random Forest

## 參數設定
- CatBoost
    - cat_features = [2,3,6,7] (string feature的column indices)
    - class_weights = [1,200] (設定權重)
- LightGBM
    - n_estimators = 100 (boosting iterations的次數)
    - scale_pos_weight = 300 (設定權重)
- Neural Network
    - layer settings
        - 針對類別資料: 先各自embedding成10維，接著concat在一起，經過reshape之後，再Dense成10維
        - 針對數值資料: 先Dense成20維，再Dense成10維
        - 統整階段: 將上述輸出concat在一起，先Dense成10維，最後Dense成一維，進行輸出
        - activation: 除了最後一層為`sigmoid`，其他層皆使用`relu`
        - loss: binary cross entropy
        - optimizer: `adam`
    - batch size: 256
    - epochs: 10
    - class weight: {0: 1, 1: 100}
- Random Forest
    - n_estimators = 100 (boosting iterations的次數)
    - max_features = `sqrt` (決定最佳分割的feature數量)
    - random_state = 0
    - n_jobs = 8 (多核心執行)