# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('income.csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/6a8df2cf-6d43-401b-a4ef-996e200e42e1)

### data.isnull().sum()
![image](https://github.com/user-attachments/assets/6acf81cf-70f9-443f-959a-5f1a248d9e6f)

### missing=data[data.isnull().any(axis=1)]
![image](https://github.com/user-attachments/assets/07a19e6e-ae40-441e-ae56-78c34842f4e3)

### data2=data.dropna(axis=0)
![image](https://github.com/user-attachments/assets/1aa2e131-26da-400a-9d30-ce1df85ccdc1)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/d8a2a878-27c5-41f1-968a-30276acc57f9)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/3c04bdfb-3b30-4fbb-bcaa-36018ffe85b1)

### data2
![image](https://github.com/user-attachments/assets/e30c86ad-8245-47d9-9b78-8f08cbcb182d)

### new_data=pd.get_dummies(data2, drop_first=True)
![image](https://github.com/user-attachments/assets/50afdc8d-9f0f-4f5c-955b-274c76081d1c)

### columns_list=list(new_data.columns)
![image](https://github.com/user-attachments/assets/bfe606dc-94e5-44cd-b082-625b3aa756a8)

### feature=list(set(columns_list)-set(['SalStat']))
![image](https://github.com/user-attachments/assets/0b9753ae-f854-441a-9926-c8360ec38a9b)

### y=new_data['SalStat'].values
![image](https://github.com/user-attachments/assets/c23e225e-6460-4dbe-8fc9-5ae39f2d8981)

### x=new_data[feature].values
![image](https://github.com/user-attachments/assets/2d989faf-1e94-4388-8497-77a0f9160b9f)

### train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

### KNN_classifier=KNeighborsClassifier(n_neighbors=5)

```
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/92e5c2e6-c863-4f33-9a21-ac63a96d5855)

### prediction=KNN_classifier.predict(test_x)

### confusionMatrix= confusion_matrix(test_y,prediction)
![image](https://github.com/user-attachments/assets/67f54dbb-a8b8-4b8e-b88b-dbb2c338da20)

### accuracy_score=accuracy_score(test_y,prediction)
![image](https://github.com/user-attachments/assets/54c8d8e1-2a7c-4fe6-8317-540a5055716e)

### data.shape
![image](https://github.com/user-attachments/assets/7ec222a5-d8a5-497b-aeb1-cd6e9bd08c32)


# RESULT:
Thus feature scaling and selection is performed.
