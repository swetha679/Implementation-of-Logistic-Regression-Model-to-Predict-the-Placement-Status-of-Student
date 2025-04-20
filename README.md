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
# HEAD

![Screenshot 2025-04-19 173322](https://github.com/user-attachments/assets/acae768c-b38c-45a9-95d2-913b721fe6ab)


# COPY:

![Screenshot 2025-04-19 173330](https://github.com/user-attachments/assets/816c2935-c0be-47ce-bb5e-dc7aa4cf32d1)



# FIT TRANSFORM

![Screenshot 2025-04-19 173341](https://github.com/user-attachments/assets/91a05e2a-b924-4439-a828-b84cfe63b589)


#  LOGISTIC REGRESSION
![Screenshot 2025-04-19 173351](https://github.com/user-attachments/assets/652c22b4-ff0e-49b5-bf38-d508e8d70e10)




#  ACCURACY SCORE
![Screenshot 2025-04-19 173401](https://github.com/user-attachments/assets/1ab04dec-bc2a-42ec-802c-ea13bff1e5c4)


# CONFUSION MATRIX
![Screenshot 2025-04-19 173409](https://github.com/user-attachments/assets/8f807836-9013-4e77-b94e-3346dd23680b)


# CLASSIFICATION REPORT

![Screenshot 2025-04-19 173422](https://github.com/user-attachments/assets/6265df4b-6ac3-40bf-8b83-e87988927827)


# PREDICTION

![Screenshot 2025-04-19 160442](https://github.com/user-attachments/assets/12868059-fcdb-4a5f-bdaa-f920c1c8e52d)



















## Result:
Thus, the program to implement the Logistic Regression Model to Predict the Placement Status of Students is written and verified using Python programming.
