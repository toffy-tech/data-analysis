import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv

iris_data=pd.read_csv("ayame.csv",encoding="utf-8")
y=iris_data.loc[:,"Name"]
x=iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,train_size=0.9,shuffle=True)
#shuffle=Trueを記述することであらかじめランダムにしてから、値をスプリットしている。
clf=SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("正解率＝",accuracy_score(y_test,y_pred))
