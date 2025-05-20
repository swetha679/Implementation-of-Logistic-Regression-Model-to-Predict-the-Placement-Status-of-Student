# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the Logistic Regression Model to predict the Placement Status of students.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression, find the predicted values of accuracy and confusion matrices.
5. Display the results.

## Program:
```

Program to implement the Logistic Regression Model to Predict the Placement Status of Students.
Developed by: B R SWETHA NIVASINI
RegisterNumber:212224040345
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
x
print(x)
y=data1["status"]
y
print(y)
print()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()


from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![The Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![image](https://github.com/user-attachments/assets/ca0f6616-e1c9-4474-b221-0542ad1fb076)


![image](https://github.com/user-attachments/assets/808a0582-0f2c-4b39-8fff-02639ebf1f44)

![image](https://github.com/user-attachments/assets/797771f6-a330-4ccc-94ef-4938ab863d99)


![image](https://github.com/user-attachments/assets/aa98c682-74eb-40be-99ce-cd1265a9dbe0)


![image](https://github.com/user-attachments/assets/f502d7b6-d0c8-485a-98bc-1d78363df5f0)


![image](https://github.com/user-attachments/assets/96f6a06a-9411-4d52-99d2-3ec90168db93)


![image](https://github.com/user-attachments/assets/0fd052b3-68b5-429b-b889-21c49cdd943e)


![image](https://github.com/user-attachments/assets/b00ae6d9-a3f0-43ff-82b5-8cc522f7b5b4)


![image](https://github.com/user-attachments/assets/51c516b6-fc76-48cc-8cdf-2ea80f0f013a)


![image](https://github.com/user-attachments/assets/3e42bfff-006b-42bb-91e5-8d965fd99daf)



















## Result:
Thus, the program to implement the Logistic Regression Model to Predict the Placement Status of Students is written and verified using Python programming.
