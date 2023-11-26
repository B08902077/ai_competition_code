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

# 先決定是否保留string data，接著設定不想考慮到的column
string_discard = False
drop_lst = ['txkey', 'insfg', 'csmam', 'chid', 'cano', 'mchno', 'acqic'] if string_discard else ['txkey', 'insfg', 'csmam']

# read data
# missing value: 不處理
df_LGBM = pd.concat([pd.read_csv('training.csv').drop(drop_lst, axis = 1), pd.read_csv('public.csv').drop(drop_lst, axis = 1)], axis=0)
test_LGBM = pd.read_csv('private_1_processed.csv')
test_txkey = test_LGBM.txkey.values.reshape(-1)
test_LGBM = test_LGBM.drop(drop_lst, axis=1)

# label encoding
for col in df_LGBM.columns:
    if df_LGBM[col].dtype == object:
        mapper = {v: k for k, v in enumerate(pd.unique(pd.concat([df_LGBM[col], test_LGBM[col]], axis=0)))}
        df_LGBM[col] = df_LGBM[col].map(mapper).astype('category')
        test_LGBM[col] = test_LGBM[col].map(mapper).astype('category')

# 設定train data和test data
X_LGBM = df_LGBM.drop('label', axis = 1).values
y_LGBM = df_LGBM.label.values.astype('int')
X_test_LGBM = test_LGBM.values

# train and predict
clf_LGBM.fit(X_LGBM, y_LGBM)
prediction_LGBM = clf_LGBM.predict(X_test_LGBM)

# 將預測結果寫入檔案
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_LGBM})
df_result.to_csv(f'result_LGBM{"_NoString" if string_discard else ""}.csv', index=False)