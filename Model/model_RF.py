import numpy as np
import pandas as pd

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=100,
                                max_features='sqrt',
                                random_state=0,
                                n_jobs=8)

# 先決定是否保留string data，接著設定不想考慮到的column
string_discard = False
drop_lst = ['txkey', 'insfg', 'csmam', 'chid', 'cano', 'mchno', 'acqic'] if string_discard else ['txkey', 'insfg', 'csmam']

df_RF = pd.concat([pd.read_csv('training.csv').drop(drop_lst, axis = 1), pd.read_csv('public.csv').drop(drop_lst, axis = 1)], axis=0)
test_RF = pd.read_csv('private_1_processed.csv')
test_txkey = test_RF.txkey.values.reshape(-1)
test_RF = test_RF.drop(drop_lst, axis = 1)

# missing value: 全部替換成 -1
df_RF.fillna(-1, inplace = True)
test_RF.fillna(-1, inplace = True)

# handle string data if needed (label encoding)
if not string_discard:
    cat_cols = ['chid', 'cano', 'mchno', 'acqic']
    for col in cat_cols:
        mapper = {v: k for k, v in enumerate(pd.unique(pd.concat([df_RF[col], test_RF[col]], axis=0)))}
        df_RF[col] = df_RF[col].map(mapper).astype('category')
        test_RF[col] = test_RF[col].map(mapper).astype('category')

# 設定train data和test data
y_RF = df_RF.label.values.reshape(-1) # 1d array
x_RF = df_RF.drop('label', axis = 1).values
X_test_RF = test_RF.values

# train and predict
clf_RF.fit(x_RF, y_RF)
prediction_RF = clf_RF.predict(X_test_RF)

# 將預測結果寫入檔案
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_RF})
df_result.to_csv(f'result_RF{"_NoString" if string_discard else ""}.csv', index=False)