
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score

df = pd.read_csv("C:\\Users\\afnan\Downloads\\bmi.csv")

X = df[["Height","Weight"]]
Y = df[["Index"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

filename = 'bmipredictor.sav'
pickle.dump(model, open(filename, 'wb'))

