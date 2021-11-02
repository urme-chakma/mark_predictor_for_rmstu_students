#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv('rmstu_student_info.csv')
#print("The dataframe is here:\n",df)

#print(df.head())
#print(df.tail())
#print(df.shape)

#Discover and visualize the data
#print("data information:\n", df.info())
#print(df.describe())


plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
#plt.show()


#Preparing the data for Machine Learning algorithms
#data cleaning
df.isnull().sum(0)

df2 = df.fillna(df.mean())
#print(df2.head())

#split dataset
X = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")
#print("shape:_____________________")
#print("shape of X = ", X.shape)
#print("shape of y = ", y.shape)
#print("======================================")
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=51)
#print("Splited train and test data : ___________________")
#print("shape of X_train = ", X_train.shape)
#print("shape of y_train = ", y_train.shape)
#print("shape of X_test = ", X_test.shape)
#print("shape of y_test = ", y_test.shape)


#selecting linear regression model and training it
# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
 
lr.fit(X_train,y_train)

#print("coff : ",lr.coef_)
#print("Intercept : ", lr.intercept_)


#used algorithm
#m = 3.93
#c = 50.44
#y  = m * 4 + c 
#print("y = ", y)


lr.predict([[4]])[0][0].round(2)

y_pred  = lr.predict(X_test)
#print(y_pred)

#print(pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"]))
 
#Fine-tune your model
#print("Accuracy lavel in 1: ",lr.score(X_test,y_test))

plt.scatter(X_train,y_train)
plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")


import joblib
joblib.dump(lr, "rmstu_student_mark_predictor.pkl")
model = joblib.load("rmstu_student_mark_predictor.pkl")
study_hour = float(input("Enter your study hour: "))
print("Predicted mark: ",model.predict([[study_hour]])[0][0])