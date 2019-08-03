import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
#%%
# CSV 파일을 데이터프레임으로 업로드
subway_in = pd.read_csv(r'C:\Users\anjoon\Desktop\miniproject2\data\subway_in_final.csv', encoding = 'utf-8')
subway_in = subway_in.loc[:,['역명','지가','면적(㎡)','카테고리','임대료','승객 수']]
#%%
# target과 feature데이터로  나눔
# get_dummies로 범주형 변수를 원-핫 인코딩 처리
y_target = subway_in['임대료']
X_features = subway_in.drop(['임대료'], axis=1, inplace=False)
X_features_ohe = pd.get_dummies(X_features)
#%%
# y_target값의 분포를 정규분포로 만들기 위해서 로그 값을 취해줌
y_target_log = np.log1p(y_target)
#%%
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.2, random_state = 42)
#%%
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[1000],
    'max_depth':[6, 8, 10, 12, 14, 16, 18, 20],
    'num_leaves':[32, 48, 64, 80],
    'learning_rate':[0.01, 0.05, 1]
}
#%%
lgbm_reg = LGBMRegressor()
grid_cv = GridSearchCV(lgbm_reg, param_grid=params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_cv.fit(X_train, y_train)
# 평가지표를 MSE(오차제곱의 평균값)를 이용함, 이값이 낮아야 좋음
print('최적 하이퍼 파라미터:', grid_cv.best_params_)
print('최고 예측 점수:', -1*grid_cv.best_score_)
#%%
# 최적 파라미터 값으로 모델을 다시 수행
# 평가지표로 R2값을 사용했으며 1과 가까워야 좋음
# 제대로된 평가를 위해 np.expm1을 사용해 역로그를 취함
from lightgbm import LGBMRegressor
lgbm_reg1 = LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=18, num_leaves=48)
lgbm_reg1.fit(X_train, y_train)
lgbm_reg1.score(X_test,y_test)

#%%
import pickle
lgbmFile=open('lgbm_reg1.pckl','wb')
pickle.dump(lgbm_reg1, lgbmFile)
lgbmFile.close()