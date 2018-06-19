'''Predicting Survival in the Titanic Data Set
We will be using a decision tree to make predictions about the Titanic data set from
Kaggle. This data set provides information on the Titanic passengers and can be used to
predict whether a passenger survived or not.
Loading Data and modules
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report
Url=
https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic
-train.csv
titanic = pd.read_csv(url)
titanic.columns =
['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','E
mbarked']
You use only Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard),
and Fare to predict whether a passenger survived.

NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​and​'''


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url,usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Survived'])

#Separate out the depenent and independent features
X=titanic.iloc[:,1:].values
y=titanic.iloc[:,0].values

#Taking Care of Missing data
from sklearn.preprocessing import Imputer
imputer= Imputer()

#fit and transform in a single line
X[:,[2]]=imputer.fit_transform(X[:,[2]])
X[:,2]=X[:,2].astype(int)

#Encoding the categorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= .20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Create Decision tree model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

##Fitting RF clasiification to the trainaing set
#from sklearn.ensemble import RandomForestClassifier
#classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)
#y_pred1 = classifier.predict_proba(X_test)
#rouded=np.round(y_pred1)


cm= confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
class_report= classification_report(y_test,y_pred)

