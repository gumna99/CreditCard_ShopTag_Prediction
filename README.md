# CreditCard_ShopTag_Prediction
## predict top-3 of shop tag

[玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦](https://tbrain.trendmicro.com.tw/Competitions/Details/18)


**預測說明**
- 預測每位顧客下個月份消費金額前三名的消費類別排序。
- 需預測的消費類別包含16種類，請僅針對這16類進行預測回傳。

**問題描述**
- 我們獲得了每日的歷史消費數據（時間序列）。
- 需要注意的是，顧客和類別每個月都有輕微的變化。 因此創建一個能夠處理這種情況的健壯模型是挑戰中的一部分。

  當作「未來消費金額預測」問題：
  預測每位顧客的每種類別，當月的消費金額是多少，再取前三高的類別

---

## DATA Description
#### 消費數據
- **22130578筆**
- **消費月份 – 1~24**
- **顧客編號 – 10,000,000~10,500,000**
- **消費類別**
- **消費次數**
- **消費金額（經過神秘轉換）**
#### 消費習慣
- 國內外、線上實體、消費次數
- 卡片1-14, other 消費次數
- 卡片1-14, other 消費金額佔比
#### 消費特徵
- 基本資訊:婚姻、學歷、行業、國籍、職位……. (遺漏值太多，暫不考慮)

---

## Steps
1. 查看資料、且過濾出需要的16種類別
2. 探索性因素分析
    - 進行統計分析圖，檢視數據分布
    - 異常值、空值排除
3. 特徵工程 -- 建立時間序列training/validation data
4. Model training >>XGBOOST or others
5. Predict
6. 取top-3

---

## exploration analysis
![](https://github.com/gumna99/CreditCard_ShopTag_Prediction/blob/master/exploration_analysis.png)
