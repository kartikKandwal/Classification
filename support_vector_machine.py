#supporting_vector_machine
'''
1.import library
2.import dataset
3.splitting data set
4.feature scaling
5.training support vectore machine
6.prediction a new result
7.prediction test set result
8.make confusion matrix
9.visualising the training set result
10.visualising the test set
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d1=pd.read_csv("Social_Network_Ads.csv")
X=d1.iloc[:,:-1].values
y=d1.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.svm import  SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)

print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
t=accuracy_score(y_test, y_pred)
print(t)