"""
統合不同模型的預測結果，進行多數決投票，並將最終的預測結果寫入result.csv
"""
import numpy as np
import pandas as pd

string_discard = False
predictions, test_txkey = dict(), list()
for model_name in ['RF', 'CB', 'LGBM', 'NN']:
    csv_name = f'./Model/result_{model_name}{"_NoString" if string_discard else ""}.csv'
    df_temp = pd.read_csv(csv_name)
    if model_name == 'RF':
        test_txkey = df_temp.txkey.values.reshape(-1)
    predictions[model_name] = df_temp.pred.values.reshape(-1)

vote_candidates = ['RF', 'CB', 'LGBM', 'NN']
vote_thres = 3
prediction_vote = np.where(np.array([predictions[model_name] for model_name in vote_candidates]).sum(axis=0) >= vote_thres, 1, 0)
df_result = pd.DataFrame({'txkey': test_txkey, 'pred': prediction_vote})
df_result.to_csv(f'result.csv', index=False)