import numpy as np
import pandas as pd

# CatBoost
from catboost import CatBoostClassifier
clf_CB = CatBoostClassifier(cat_features=[2,3,6,7],
                            silent = True,
                            class_weights = [1,200])

# 先決定是否保留string data，接著設定不想考慮到的column
string_discard = False
drop_lst = ['txkey', 'insfg', 'csmam', 'chid', 'cano', 'mchno', 'acqic'] if string_discard else ['txkey', 'insfg', 'csmam']

# read data
df_CB = pd.concat([pd.read_csv('training.csv').drop(drop_lst, axis = 1), pd.read_csv('public.csv').drop(drop_lst, axis = 1)], axis=0)
test_CB = pd.read_csv('private_1_processed.csv')
test_txkey = test_CB.txkey.values.reshape(-1)
test_CB = test_CB.drop(drop_lst, axis=1)

# missing value: string data的column替換成字串 'NAN' ，其他的column則替換成數字 -1
for col in df_CB.columns:
    if df_CB[col].dtype == object:
        df_CB[col].fillna('NAN', inplace=True)
    else:
        df_CB[col] = df_CB[col].fillna(-1).astype(int)
for col in test_CB.columns:
    if test_CB[col].dtype == object:
        test_CB[col].fillna('NAN', inplace=True)
    else:
        test_CB[col] = test_CB[col].fillna(-1).astype(int)

# 設定train data和test data
X_CB = df_CB.drop('label', axis=1).values
y_CB = df_CB.label.values.reshape(-1) # 1d array
X_test_CB = test_CB.values

# train and predict
clf_CB.fit(X_CB, y_CB)
prediction_CB = clf_CB.predict(X_test_CB)

# 將預測結果寫入檔案
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_CB})
df_result.to_csv(f'result_CB{"_NoString" if string_discard else ""}.csv', index=False)