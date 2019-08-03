
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from flaskext.mysql import MySQL

data1 = pd.read_csv(r'C:\Users\CPB06GameN\project\subway_in.csv')
data2 = pd.read_csv(r'C:\Users\CPB06GameN\project\train_in_df.csv')
#%% 데이터합치기
data=data1.loc[:,['역명','지역(구)','지역(동)','지가','면적(㎡)','카테고리','임대료']]
data2=pd.DataFrame(data2['승객 수'])
new_data=pd.concat([data,data2],axis=1)
new_data1=new_data.dropna(axis=0)
new_data1.isnull().sum()
#%%
aaa=list(new_data1['임대료'].values)
bbb=[]
for i in range(len(aaa)):
    bbb.append(int(aaa[i].replace(',','')))
new_data1['임대료']=bbb
data_dummies=pd.get_dummies(new_data1) #자동으로 one-hot encoding을 해준다
X=data_dummies.drop('임대료',axis=1)
y=data_dummies.loc[:,'임대료']
#%%
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle

#gradient boosting regression 모델구축
X_new,y_new = shuffle(X, y, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset] #일종의 train_set,test_set으로 나누는 코드
X_test, y_test = X[offset:], y[offset:]
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
gradient = ensemble.GradientBoostingRegressor(**params)
gradient.fit(X_train,y_train)
print("gradient:{:.3f}".format(gradient.score(X_test,y_test)))
print(X_train)

#%%
import pickle
gradientFile=open('gradient.pckl','wb') #w:write, b=binary 
#저장을 할 목적으로 열었다
pickle.dump(gradient, gradientFile)
#저장하겠다
gradientFile.close()