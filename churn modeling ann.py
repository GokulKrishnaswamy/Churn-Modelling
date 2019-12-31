# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Removing features based on domain knowledge
dataset.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l1 = LabelEncoder()
dataset['Geography'] = l1.fit_transform(dataset['Geography'])
l2 = LabelEncoder()
dataset['Gender'] = l2.fit_transform(dataset['Gender'])
o2 = OneHotEncoder(categorical_features = [1])
dataset = o2.fit_transform(dataset).toarray()

#Avoiding dummy variable trap
X=dataset[:,1:-1]
y=dataset[:,-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Compute class weight for balancing dataset
from sklearn.utils import class_weight
c=class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(optimizer):
    # Initialising the ANN
    classifier = Sequential()    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))  
    # Adding the second hidden layer
    #classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))  
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))    
    # Compiling the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])    
    return classifier

# Hyperparametertuning
model=KerasClassifier(create_model)
d=dict(batch_size=[64,128],epochs=[200,500],optimizer=['adam','rmsprop','sgd'])
grid=GridSearchCV(model,d)
grid.fit(X_train,y_train)
print(grid.best_params_)

# Fitting the ANN to the Training set
classifier=create_model(grid.best_params_['optimizer'])
classifier.fit(X_train, y_train, batch_size = grid.best_params_['batch_size'], epochs = grid.best_params_['epochs'],class_weight=c)

# Predicting the Test set results
y_score = classifier.predict(X_test)

# Choose optimal threshold from roc curve
from sklearn.metrics import roc_curve
fpr,tpr,threshold = roc_curve(y_test,y_score)
plt.plot(fpr,tpr)
plt.show()

# Predicting classes, choose a threshold according to business requirement
y_pred = (y_score > 0.5)

# Evaluating model performance
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import precision_score
print(precision_score(y_test,y_pred))

from sklearn.metrics import recall_score
print(recall_score(y_test,y_pred))