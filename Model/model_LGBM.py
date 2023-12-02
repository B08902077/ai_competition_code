"""
使用LightGBM模型進行預測，並將預測結果寫入檔案
"""
import numpy as np
import pandas as pd

# LightGBM
from lightgbm.sklearn import LGBMClassifier
clf_LGBM = LGBMClassifier(n_estimators=100,
                          scale_pos_weight=300,
                          verbosity=-1) # only fatal problems are shown

# 設定不想考慮到的column
drop_lst = ['txkey', 'insfg', 'csmam']

# read data
# missing value: 不處理
df_LGBM = pd.concat([pd.read_csv(filename).drop(drop_lst, axis = 1) for filename in ['training.csv', 'public.csv', 'private_1.csv']], axis=0)
test_LGBM = pd.read_csv('private_2_processed.csv')
test_txkey = test_LGBM.txkey.values.reshape(-1)
test_LGBM = test_LGBM.drop(drop_lst, axis=1)

# label encoding
for col in df_LGBM.columns:
    if df_LGBM[col].dtype == object:
        mapper = {v: k for k, v in enumerate(pd.unique(pd.concat([df_LGBM[col], test_LGBM[col]], axis=0)))}
        df_LGBM[col] = df_LGBM[col].map(mapper).astype('category')
        test_LGBM[col] = test_LGBM[col].map(mapper).astype('category')
        
cat_features = ['contp','etymd','mcc','ecfg','bnsfg','stocn','scity','stscd','ovrlt','flbmk','hcefg','csmcu','flg_3dsmk']
df_LGBM[cat_features] = df_LGBM[cat_features].astype('category')
test_LGBM[cat_features] = test_LGBM[cat_features].astype('category')

# 設定train data和test data
X_LGBM = df_LGBM.drop('label', axis = 1).values
y_LGBM = df_LGBM.label.values.astype('int')
X_test_LGBM = test_LGBM.values

# train and predict
clf_LGBM.fit(X_LGBM, y_LGBM)
prediction_LGBM = clf_LGBM.predict(X_test_LGBM)

# 將預測結果寫入檔案
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_LGBM})
df_result.to_csv(f'result_LGBM.csv', index=False)