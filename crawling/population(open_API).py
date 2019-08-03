# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:51:47 2019

@author: anjoon
"""

import pandas as pd
from bs4 import BeautifulSoup
import requests
#%%
startnumber=1
endnumber=1000 #한번에 1000개까지만 데이터를 가져올 수 있기 때문에 설정
year_list = []
address_list = []
population_list = []

#%% 2018
#한번에 1000개까지 가져오므로 4000까지 하면 총 4번을 돌리는 꼴!
while endnumber <= 4000:
    url= 'http://openapi.seoul.go.kr:8088/586e4754636a616a363874614d716f/xml/VwsmTrdarFlpopQq/'+str(startnumber)+'/'+str(endnumber)+'/2018'
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    
    #stdr_yy_cd(year의 xml상 출력값)인 것을 모두 찾아서 year에 넣는다
    year = soup.find_all('stdr_yy_cd')
    address = soup.find_all('trdar_cd_nm')
    population = soup.find_all('tot_flpop_co')
    
    for code in year:
        year_list.append(code.text)
    for code in address:
        address_list.append(code.text)
    for code in population:
        population_list.append(code.text)
    startnumber+=1000
    endnumber+=1000

    
#%%
#위에서 담은 리스트를 가지고 데이터프레임을 만듦 
data={}
data['Year'] = year_list
data['Address'] = address_list
data['Population'] = population_list

df = pd.DataFrame(data)    
df.to_csv(r"C:\Users\anjoon\Desktop\mini2\crawling\population.csv",header=False,index=False)
