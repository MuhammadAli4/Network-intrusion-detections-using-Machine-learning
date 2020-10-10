# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset_train = pd.read_csv('Train_data.csv')
dataset_test = pd.read_csv('Test_data.csv')
X = dataset_train.iloc[:, :-1]
X_testing = dataset_test
y = dataset_train[dataset_train.columns[-1]]
y = pd.DataFrame({'class':y.values})


#One Hot encoder
w1 = pd.get_dummies(X['protocol_type'], prefix='protocol_type', drop_first=True)
w2 = pd.get_dummies(X['service'], prefix='service', drop_first=True)
w3 = pd.get_dummies(X['flag'], prefix='flag', drop_first=True)

#Drop and add rows
X = X.drop(['protocol_type', 'service', 'flag'], axis=1)
X = pd.concat([w1,w2,w3, X], axis=1)

# Analyse corelations
y_hot = pd.get_dummies(y['class'], prefix='class', drop_first=True)
corr_matrix = pd.concat([X, y_hot], axis=1)
corr_matrix = corr_matrix.corr(method ='pearson')
#corr_matrix.to_csv('corr_metrics.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True, random_state=4)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Random Forest Classification to the Training set
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state =0)
#classifier.fit(X_train, y_train)


# Predicting the Test set results
#y_pred = classifier.predict(X_test)
#y_pred_train = classifier.predict(X_train)

#changing y
y_train = pd.DataFrame(y_train).to_numpy()
y_test = pd.DataFrame(y_test).to_numpy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Part 2 - Building the ANN
import tensorflow as tf
tf.__version__
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)


# Evaluating performance

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Data_Balance = y.value_counts()
#DataTest_Balance = y_test.value_counts()
Test_accuracy = accuracy_score(y_test, y_pred)*100
Test_precision = precision_score(y_test, y_pred, pos_label='anomaly')
Test_recall = recall_score(y_test, y_pred, pos_label='anomaly')
Test_F1 = f1_score(y_test, y_pred, pos_label='anomaly')

print(Test_accuracy, Test_precision, Test_recall, Test_F1)





