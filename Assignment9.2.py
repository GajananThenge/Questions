'''In this assignment students will build the random forest model after normalizing the
variable to house pricing from boston data set.
Following the code to get data into the environment:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target

NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​and​'''

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets
    from sklearn.metrics import confusion_matrix,accuracy_score
    boston = datasets.load_boston()
    features = pd.DataFrame(boston.data, columns=boston.feature_names)
    targets = boston.target
    
    sc_y=StandardScaler()
    targets = sc_y.fit_transform(targets.reshape(-1,1)).astype(int)
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test= train_test_split(features,targets,test_size= .20,random_state=0)
    
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    
    #Fitting RF clasiification to the trainaing set
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    y_pred1 = classifier.predict_proba(X_test)
    rouded=np.round(y_pred1)
    acc=accuracy_score(y_test,y_pred)
    cm= confusion_matrix(y_test,y_pred)
    
except Exception as e:
    print(e)