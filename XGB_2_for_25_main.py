#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set() # 使用初始网格状绘图纸
# myfont = matplotlib.font_manager.FontProperties(fname="../input/fontssimhei/simhei.ttf")
# plt.rcParams['font.family'] = ['Times New Roman']


# In[ ]:


data = pd.read_csv("tbrain_cc_training_48tags_hash_final.csv")


# In[ ]:


pd.set_option("display.max_columns", None)
data.head(5).append(data.tail(5))


# In[ ]:


#训练集数据形状
print(data.shape)


# # 過濾出關心的16種類的資料
# 2,6,10,12,13,15,18,19,21,22,25,26,36,37,39,48

# In[5]:


data_16 = data.query( "(shop_tag == '2' | shop_tag == '6'| shop_tag == '10' | shop_tag == '12'| shop_tag == '13'| shop_tag == '15'| shop_tag == '18'| shop_tag == '19'| shop_tag == '21'| shop_tag == '22'| shop_tag == '25'| shop_tag == '26'| shop_tag == '36'| shop_tag == '37'| shop_tag == '39'| shop_tag == '48')")


# In[14]:


print(data_16.groupby('shop_tag').size())
print(data_16.groupby(['chid','dt','shop_tag']).size())


# # 探索性分析

# In[65]:


# shoptag 銷售種類
fig = plt.figure(figsize=(20,10)) # 画布大小设置
plt.subplots_adjust(hspace=.4) # 调整子图的位置距离，hspace（垂直间距）= 0.4

#查看shop_tag消費類別的分布

plt.subplot2grid((3,3), (0,0), rowspan=1, colspan=3)  # 建立3*3的网格区域，当前位置为0行0列，行方向占1个单位，列方向占据3个单位
data16_c5['shop_tag'].value_counts(normalize=True).plot(kind='bar', color='orangered') # 使用柱状图
plt.title('shop_tag消費類別分布圖')
plt.xlabel('shop_tag')
plt.ylabel('times')


# In[69]:


# 月份
fig = plt.figure(figsize=(20,10)) # 画布大小设置
plt.subplots_adjust(hspace=.4) # 调整子图的位置距离，hspace（垂直间距）= 0.4

#查看date_block_num的分布
plt.subplot2grid((3,3), (2,0), rowspan=1, colspan=3)  # 建立3*3的网格区域，当前位置为2行0列，行方向占1个单位，列方向占据3个单位
data16_c5['dt'].value_counts(normalize=True).plot(kind='bar', color='darkseagreen') # 使用柱状图
plt.title('dt')
plt.xlabel('dt')
plt.ylabel('times')

plt.show()


# In[12]:


# txn_amt=消費金額
fig = plt.figure(figsize=(20,10)) # 画布大小设置
plt.subplots_adjust(hspace=.4) # 调整子图的位置距离，hspace（垂直间距）= 0.4

plt.subplot2grid((3,3), (1,0), rowspan=1, colspan=3)  # 建立3*3的网格区域，当前位置为1行0列，行方向占1个单位，列方向占据1个单位
data_16['txn_amt'].plot(kind='hist', color='darkorange')  # 使用直方图
plt.title('txn_amt')
plt.xlabel('txn_amt')
plt.ylabel('times')


# In[ ]:


print(data16_c5_Goup.get_group)


# # (1)處理異常銷售金額值(top,tail)

# In[3]:


#查看排序最大的前五个价格
data16['txn_amt'].sort_values(ascending=False)[:5]


# In[78]:


pd.set_option("display.max_columns", None) # 顯示所有欄位

data16.iloc[8022597:8022598,:]


# In[4]:


#查看所有39類別，有無其他異常值
#8022597 金額特別高故刪除(?
data16[(data16.shop_tag == 39) & (data16.txn_amt>2.533508e+25)]
data16_c = data16.drop(index=[8022597])


# In[ ]:


# gg = data16[(data16.shop_tag == 39) & (data16.txn_amt>2.533508e+25)]
# print(data16[(gg)])

gg = data_16[(data_16.shop_tag == 39) & (data_16.txn_amt>2.533508e+25)]
print(data_16[(gg)])


# In[112]:


data16_c.iloc[8022597:8022598,:]


# In[103]:


#检查极小值端的异常情况
data16['txn_amt'].sort_values(ascending=True)[:5850]


# In[99]:


data16[ (data16.txn_amt < 22.370852)]


# # 小結論
# ## 刪除最大值Index=8022597
# ## 無異常最小值

# # 查看排序最大的前五个商品销售数量
# # 查看排序最小的前五个商品销售数量

# In[155]:


#查看排序最大的前五个商品销售数量
data16_c['txn_cnt'].sort_values(ascending=False)[:10]


# In[124]:


#1618504
data16_c.iloc[1618504:1618505,:]


# In[136]:


#查看该商品对应的销售量信息
#類別 36
sale_num = data16_c[(data16_c.shop_tag == 36)]
print(sale_num)
sale_num["txn_cnt"].describe()


# In[154]:


data16_c[(data16_c.shop_tag == 36) & (data16_c.chid == 10132459)]


# In[137]:


#21286266 
data16_c.iloc[21286265:21286266,:]


# In[139]:


#查看该商品对应的销售量信息
sale_num = data16_c[(data16_c.shop_tag == 10)]
# print(sale_num)
sale_num["txn_cnt"].describe()


# In[156]:


data16_c[(data16_c.shop_tag == 10) & (data16_c.chid == 10436426)]


# In[123]:


data16_c.iloc[10469593:10469594,:]


# # 無刪除最大數量 >> 不造成異常消費額的高低

# In[146]:


#查看排序最大的後五个商品销售数量
data16_c['txn_cnt'].sort_values(ascending=True)[:5]


# In[147]:


data16_c.iloc[12695976:12695977,:]


# In[149]:


data16_c.iloc[3516422:3516423,:]


# # ---

# # 笛卡爾乘積

# In[2]:


data16_c = pd.read_csv("data16_c.csv")


# In[8]:


data16_c5 = data16_c.iloc[:,0:5]
data16_c5


# In[4]:


get_ipython().run_cell_magic('time', '', "#將所有24個月份進行相同的數據合併df建立\nfrom itertools import product \nchid = np.arange(10000000,10500000)\nshop_tag = [2,6,10,12,13,15,18,19,21,22,25,26,36,37,39,48]\n\nmonths = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]\n# print(months.shape)\ncartesian = []\nfor month in months:\n    cartesian.append(np.array(list(product(*[chid, shop_tag, [month]])), dtype='int32'))\n    \n    \n#     shops_in_month = sales_train.loc[sales_train['date_block_num']==month, 'shop_id'].unique()\n#     items_in_month = sales_train.loc[sales_train['date_block_num']==month, 'item_id'].unique()\n#     cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))\n    \ncartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['chid', 'shop_tag', 'dt'], dtype=np.int32)")


# 合併兩個DF
# cartesian_df & data16_c5

# In[9]:


#pd.merge()方法进行合并连接，left表示只保留左边的主键，只在右边主键中存在的行就不取了
new_train = pd.merge(cartesian_df, data16_c5, on=['chid', 'shop_tag', 'dt'], how='left').fillna(0) 


# In[12]:


new_train.sort_values(['dt','chid','shop_tag'], inplace = True)
new_train.head()
new_train.shape


# In[ ]:


#删除系统中不需要的列表，釋放內存
# del x
# del x2
# del cartesian_df
# del cartesian
# del chid
# del data16 


# # 生成滯後特徵和平均編碼

# In[16]:


#定义滞后特征添加函数
def generate_lag(train, months, lag_column):
    for month in months:
        # 创建滞后特征
        train_shift = train[['dt','chid','shop_tag', lag_column]].copy()
        train_shift.columns = ['dt','chid','shop_tag', lag_column+'_lag_'+ str(month)]
        train_shift['dt'] += month
        #新列表连接到训练集中
        train = pd.merge(train, train_shift, on=['dt','chid','shop_tag'], how='left')
        del train_shift
    return train


# In[19]:


#定义向下数据类型转变函数，作用是将float64类型转变成float16，将int64转变成int16
# （用于缩减内存量,否则后续无法运行）
from tqdm import tqdm_notebook   # 进度读取条使用
def downcast_dtypes(df):   
    # 选择需要处理的列 
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    #开始数据转换
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)
    
    return df


# In[22]:


#使用变换函数来更数据类型
new_train2 = downcast_dtypes(new_train)  


# In[23]:


new_train2.shape


# In[26]:


del new_train


# In[37]:


# %%time
# #增加目标变量-消費次數的滞后特征
# new_train3 = generate_lag(new_train2, [1,2,3,4,5,6,12], 'txn_cnt')
# # print(new_train3)


# In[27]:


get_ipython().run_cell_magic('time', '', "#增加目标变量-消費金額的滞后特征\nnew_train3 = generate_lag(new_train2, [1,2,3,4,5,6,12], 'txn_amt')\n# print(new_train3)")


# In[ ]:


# new_train4 = pd.read_csv('new_train4.csv')


# ### chid_txn_amt_mean

# In[35]:


group = new_train.groupby(['dt', 'chid'])['txn_amt'].mean().rename('chid_txn_amt_mean').reset_index()
print(group)


# In[36]:


new_train3 = pd.merge(new_train3, group, on=['dt', 'chid'], how='left')
print(new_train3)


# In[ ]:





# In[37]:


get_ipython().run_cell_magic('time', '', "new_train3 = generate_lag(new_train3, [1,2,3,6,12], 'chid_txn_amt_mean')\n#删除不需要的'item_month_mean'属性\nnew_train3.drop(['chid_txn_amt_mean'], axis=1, inplace=True)")


# In[7]:


new_train4.to_csv('new_train4_1.csv',index=0)


# In[8]:


del group
del new_train4


# ### shop_tag_txn_amt_mean

# In[ ]:


group = new_train.groupby(['dt', 'shop_tag'])['txn_amt'].mean().rename('shop_tag_txn_amt_mean').reset_index()
print(group)
new_train3 = pd.merge(new_train3, group, on=['dt', 'chid'], how='left')
print(new_train3)

new_train3 = generate_lag(new_train3, [1,2,3,6,12], 'shop_tag_txn_amt_mean')
#删除不需要的'item_month_mean'属性
new_train3.drop(['shop_tag_txn_amt_mean'], axis=1, inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#增加目标变量-消費金額的滞后特征\n#按"月份"和"chid"排序并取其月销量的均值\n\n# group = new_train4.groupby([\'dt\', \'chid\'])[\'txn_amt\'].mean().rename(\'chid_txn_amt_mean\').reset_index()\n# #将新表添加到new_train的右侧，对应\'date_block_num\', \'item_id\'属性\n# new_train4 = pd.merge(new_train4, group, on=[\'dt\', \'chid\'], how=\'left\')\n\n# del group\n#对[1,2,3,6,12]月进行月销量滞后添加（均值填充）\nnew_train4 = generate_lag(new_train4, [1,2,6,12], \'chid_txn_amt_mean\')\n#删除不需要的\'item_month_mean\'属性\nnew_train4.drop([\'chid_txn_amt_mean\'], axis=1, inplace=True)\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#增加目标变量-消費金額的滞后特征\n#按"月份"和"shop_tag"排序并取其月销量的均值\n\ngroup = new_train3.groupby([\'dt\', \'shop_tag\'])[\'txn_amt\'].mean().rename(\'txn_amt_mean\').reset_index()\n#将新表添加到new_train的右侧，对应\'date_block_num\', \'item_id\'属性\nnew_train3 = pd.merge(new_train3, group, on=[\'dt\', \'chid\'], how=\'left\')\n#对[1,2,3,6,12]月进行月销量滞后添加（均值填充）\nnew_train3 = generate_lag(new_train3, [1,2,3,6,12], \'txn_amt_mean\')\n#删除不需要的\'item_month_mean\'属性\nnew_train3.drop([\'txn_amt_mean\'], axis=1, inplace=True)\n\ndel group')


# # 填補null

# In[7]:


# new_train4 = new_train4.drop(columns=["chid_txn_amt_mean"])


# In[44]:


#使用0来填补，表示没有数据的样本
def fill_nan(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isna().any()):
            df[col].fillna(0, inplace=True)         
    return df


# In[49]:


new_train3_fillnan =  fill_nan(new_train3)


# In[52]:


new_train3_fillnan = new_train3_fillnan.drop(columns=['txn_cnt','chid_txn_amt_mean'])


# ## 數據集切分

# In[56]:


#训练数据的特征提取
train_feature = new_train3_fillnan[new_train3.dt < 24].drop(['txn_amt'], axis=1)
#训练数据的标签提取
train_label = new_train3_fillnan[new_train3.dt < 24]['txn_amt']
#验证数据的特征提取
val_feature = new_train3_fillnan[new_train3_fillnan.dt == 24].drop(['txn_amt'], axis=1)
#验证数据的标签提取
val_label = new_train3_fillnan[new_train3_fillnan.dt == 24]['txn_amt']

test_feature = new_train3_fillnan[new_train3_fillnan.dt == 25].drop(['txn_amt'], axis=1)


# In[5]:


train_feature.shape,train_label.shape,val_feature.shape,val_label.shape, test_feature.shape


# ### 建立XGBOOST

# In[6]:


from xgboost import XGBRegressor
#设定模型参数
model = XGBRegressor(n_estimators=3000,
                     max_depth=10)
#                      colsample_bytree=0.5, 
#                      subsample=0.5, 
#                      learning_rate = 0.03
#                     )


# In[7]:


#檢查inf nan
print(np.isnan(train_feature2).any())
print(np.isnan(train_label2).any())
print(np.isnan(val_feature2).any())
print(np.isnan(val_label2).any())
print(np.isnan(test_feature2).any())
print("------------")
print(np.isinf(train_feature2).any())
print(np.isinf(train_label2).any())
print(np.isinf(val_feature2).any())
print(np.isinf(val_label2).any())
print(np.isinf(test_feature2).any())
print("------------")

print(np.isfinite(train_feature2).all())
print(np.isfinite(train_label2).all())
print(np.isfinite(val_feature2).all())
print(np.isfinite(val_label2).all())
print(np.isfinite(test_feature2).all())


# In[ ]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_feature2_1 = scaler.fit_transform(train_feature2.iloc[:,3:])
# pd.DataFrame(train_feature2_1).iloc[30:50]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_feature2_1 = scaler.fit_transform(train_feature2.iloc[:,3:])
train_feature2_1 = pd.DataFrame(train_feature2_1)
train_feature2_1 = pd.concat([train_feature2.iloc[:,0:3],train_feature2_1])
pd.DataFrame(train_feature2_1)


# ### training

# In[ ]:


train_feature = train_feature.reset_index()
train_label = train_label.reset_index()
val_feature = val_feature.reset_index()
val_label = val_label.reset_index()


# In[ ]:


train_feature


# In[1]:


get_ipython().run_cell_magic('time', '', '#进行模型训练，并设置早停函数(建议在kaggle端进行)\nmodel.fit(train_feature.values, train_label.values, \n          eval_metric="rmse", \n          eval_set=[(train_feature.values, train_label.values), (val_feature.values, val_label.values)], \n          verbose=True, \n          early_stopping_rounds = 20)')


# In[9]:


#导出预测结果
y_pred = model.predict(test_feature.values)


# In[ ]:


#特征重要性查看
importances = pd.DataFrame({'feature':new_train.drop('txn_amt', axis = 1).columns,'importance':np.round(model.feature_importances_,3)}) 
importances = importances.sort_values('importance',ascending=False).set_index('feature') 
importances = importances[importances['importance'] > 0.01]

importances.plot(kind='bar',
                 title = 'Feature Importance',
                 figsize = (8,6),
                 grid= 'both')


# In[ ]:


y_pred


# # 倒出結果，作成取top-3 dataframe

# In[ ]:





# push to github

# In[3]:


get_ipython().system('git_status')


# In[ ]:




