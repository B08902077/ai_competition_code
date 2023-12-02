# ai_competition_code
因為我總共使用四種不同的模型進行訓練、預測，每個模型處理資料的方式不太一樣，因此我只創建Model資料夾，關於資料前處理以及訓練模型等細節，請直接參閱Model資料夾裡面的程式碼及README  
## 資料夾結構
```
.
├ Model
│ ├ model_CB.py
│ ├ model_LGBM.py
│ ├ model_NN.py
│ ├ model_RF.py
│ └ README
├ main.py
├ requirements.txt
└ README
```
## 程式碼執行
假設Model資料夾已存放了`training.csv`, `public.csv`, `private_1.csv`和 `private_2_processed.csv`等資料檔案，首先執行Model資料夾裡面的python程式:  
```
python ./Model/model_CB.py
python ./Model/model_LGBM.py
python ./Model/model_NN.py
python ./Model/model_RF.py
```
接著執行`main.py`，此程式會統合上述模型的預測結果，進行多數決投票，並將最終的預測結果寫入`result.csv`:
```
python ./main.py
```
