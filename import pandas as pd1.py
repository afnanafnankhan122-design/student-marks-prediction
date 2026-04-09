import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data={
    "age":[33,44,55,66,77],
    "study_hour":[3,4,5,6,7],
    "marks":[40,50,60,70,80]
}
df=pd.DataFrame(data)

X=df[["age","study_hour"]]
y=df["marks"]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

mid=LinearRegression()
mid.fit(x_train_scaled,y_train)
print(mid.predict(x_test_scaled))
