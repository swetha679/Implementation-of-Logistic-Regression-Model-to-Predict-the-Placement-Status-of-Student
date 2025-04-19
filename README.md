# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: B R SWETHA NIVASINI
RegisterNumber:212224040345

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
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
# HEAD


![Screenshot 2025-04-19 160215](https://github.com/user-attachments/assets/71e6ffaa-fe12-40e3-b016-35e9701dd98d)

# COPY:



![Screenshot 2025-04-19 160230](https://github.com/user-attachments/assets/f8b35070-f646-4c74-9d82-5e8a725cf104)
![Screenshot 2025-04-19 160240](https://github.com/user-attachments/assets/ba72e8ef-1a4e-42b4-95c8-05f81bd73f33)
![Screenshot 2025-04-19 160253](https://github.com/user-attachments/assets/a6663b06-191f-4efc-83b8-4cf73d1c0810)


# FIT TRANSFORM

![Screenshot 2025-04-19 160307](https://github.com/user-attachments/assets/cb518db1-13dc-45ce-b2f1-1d3ac8beed20)
![Screenshot 2025-04-19 160317](https://github.com/user-attachments/assets/ca091e46-2cea-4c15-96f3-3dc7356bbb99)

#  LOGISTIC REGRESSION

![Screenshot 2025-04-19 160327](https://github.com/user-attachments/assets/923ee9bb-d2ae-440b-91ee-d15c03e5255d)
![Screenshot 2025-04-19 160334](https://github.com/user-attachments/assets/1a27dab9-1699-499f-91a6-dcf816bb9fdb)
![Screenshot 2025-04-19 160343](https://github.com/user-attachments/assets/a713731c-d123-4aa1-a0a2-633e65cfd5dd)



#  ACCURACY SCORE

![Screenshot 2025-04-19 160353](https://github.com/user-attachments/assets/01503ec4-fa8f-46a4-b9cb-822c235e0204)
![Screenshot 2025-04-19 160401](https://github.com/user-attachments/assets/2eea4219-390e-4974-9cc9-541e69dbc1b5)

# CONFUSION MATRIX

![Screenshot 2025-04-19 160409](https://github.com/user-attachments/assets/d45b0c35-b662-443b-add5-ba7369a80655)
![Screenshot 2025-04-19 160416](https://github.com/user-attachments/assets/6e8daf5f-0b4f-4d12-b003-1869d6272517)

# CLASSIFICATION REPORT

![Screenshot 2025-04-19 160424](https://github.com/user-attachments/assets/2ec08cae-447b-4ded-bf40-247a487a8825)
![Screenshot 2025-04-19 160432](https://github.com/user-attachments/assets/01130778-691f-4629-af49-3fd9cf65cf9b)

# PREDICTION

![Screenshot 2025-04-19 160442](https://github.com/user-attachments/assets/12868059-fcdb-4a5f-bdaa-f920c1c8e52d)



















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
