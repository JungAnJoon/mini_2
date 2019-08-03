import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import pickle
from flaskext.mysql import MySQL
from pymysql.cursors import DictCursor

#%%
mysql = MySQL(cursorclass=DictCursor) # 사전 형태
#객체 만들기
app=Flask(__name__)

app.config['MYSQL_DATABASE_HOST'] = '15.164.214.53'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '1234'
app.config['MYSQL_DATABASE_DB'] = 'test'
mysql.init_app(app)
#%%

@app.route('/') #첫페이지
#index.html 불러오는 코드(첫페이지)
def index():
    return render_template("index.html")

#train_iris에서의 모델 로드하는 코드
def load_model():
    global lgbm1 #다른 곳에서도 참조할 수 있게 global형태
    lgbmFile=open("lgbm_reg1.pckl", 'rb') #rb:바이러니형태로 읽겠다
    lgbm1= pickle.load(lgbmFile) #메모리 형태로 떠있음
    lgbmFile.close()
#%%    
@app.route('/predict',methods=["POST"]) #전송방식은 고객이 입력한 결과(post)
def predict():

    station=request.form['station']
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = """select * from board where """'역명_'+station+""" = 1 """
    #try except 구문을 써서 오류날 시 예외처리를 함
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        kkk = []
        aaa=rows[0]
        bbb =list(aaa.values())
        for i in bbb:
            kkk.append(i)
        print(kkk)

        test_X=np.array(kkk)
        test_X1=test_X.reshape(1,-1)

        rent_predicted= int(lgbm1.predict(test_X1)) #test_X1으로 임대료 예측
        output= station + '  ' + '임대료: ' + str(round(np.expm1(rent_predicted)))
        return render_template("predict.html", output=output)

    except:
        outp = '일치하는 역을 찾을 수 없습니다.'
        return render_template("predict.html", outp = outp)

if __name__ =="__main__":
    load_model()
    app.run(debug=True)


