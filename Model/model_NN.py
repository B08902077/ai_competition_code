"""
使用Neural Network進行預測，並將預測結果寫入檔案
"""
import numpy as np
import pandas as pd

# 設定不想考慮到的column
drop_lst = ['txkey', 'insfg', 'csmam']

# read data
df_NN = pd.concat([pd.read_csv(filename).drop(drop_lst, axis = 1) for filename in ['training.csv', 'public.csv', 'private_1.csv']], axis=0)
test_NN = pd.read_csv('private_2_processed.csv')
test_txkey = test_NN.txkey.values.reshape(-1)
test_NN = test_NN.drop(drop_lst, axis=1)

# missing value: string data的column替換成字串 'NAN' ，其他的column則替換成數字 -1
for col in df_NN.columns:
    if df_NN[col].dtype == object:
        df_NN[col].fillna('NAN', inplace=True)
    else:
        df_NN[col] = df_NN[col].fillna(-1).astype(np.float32)
for col in test_NN.columns:
    if test_NN[col].dtype == object:
        test_NN[col].fillna('NAN', inplace=True)
    else:
        test_NN[col] = test_NN[col].fillna(-1).astype(int)

# numerical & categorical data
NN_cat_features = [2,3,4,5,6,7,8,10,12,14,15,16,17,18,19,20,21]
NN_num_features = [0,1,9,11,13]
cat_num = len(NN_cat_features)

# train
X_cat = df_NN.iloc[:, NN_cat_features].values
X_num = df_NN.iloc[:, NN_num_features].values

y_NN = df_NN.label.values.astype(np.float32)
y_NN = y_NN.reshape(len(y_NN), 1) # 2d array

# test
test_cat = test_NN.iloc[:, NN_cat_features].values
test_num = test_NN.iloc[:, NN_num_features].values

# Label Encoding
from sklearn.preprocessing import LabelEncoder

X_cat_enc, test_cat_enc = list(), list()
for i in range(X_cat.shape[1]):
    LE = LabelEncoder()
    fit_arr = np.concatenate((X_cat[:, i], test_cat[:, i]))
    LE.fit(fit_arr)

    X_cat_enc.append(LE.transform(X_cat[:, i]))
    test_cat_enc.append(LE.transform(test_cat[:, i]))

# standardization
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
SS.fit(X_num)

X_num_std = SS.transform(X_num).astype(np.float32)
test_num_std = SS.transform(test_num).astype(np.float32)

# model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, Reshape, Dropout

in_cat, em_cat = list(), list()
for i in range(len(X_cat_enc)):
    n_labels = len(np.unique(np.concatenate((X_cat_enc[i], test_cat_enc[i]))))
    in_layer = Input(shape=(1,))
    em_layer = Embedding(n_labels, 10)(in_layer)

    in_cat.append(in_layer)
    em_cat.append(em_layer)

merge_cat = concatenate(em_cat)
reshape_cat = Reshape((cat_num * 10,))(merge_cat)
dense_cat = Dense(10, activation='relu', kernel_initializer='he_normal')(reshape_cat)

in_num = Input(shape=(X_num_std.shape[-1],))
dense_num_mid = Dense(20, activation='relu', kernel_initializer='he_normal')(in_num)
dense_num = Dense(10, activation='relu', kernel_initializer='he_normal')(dense_num_mid)

merge_all = concatenate([dense_cat, dense_num])
dense_all = Dense(10, activation='relu', kernel_initializer='he_normal')(merge_all)
output = Dense(1, activation='sigmoid')(dense_all)

model = Model(inputs=[in_cat, in_num], outputs=output)

# compile, train, and predict
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit([X_cat_enc, X_num_std], y_NN,
                    epochs=3, batch_size=256, verbose=1, class_weight={0: 1, 1: 100})

prediction_NN = model.predict([test_cat_enc, test_num_std])
prediction_NN = [int(i > 0.5) for i in prediction_NN.ravel()]

# 將預測結果寫入檔案
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_NN})
df_result.to_csv(f'result_NN.csv', index=False)