import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#set up scaler
scaler = StandardScaler()

#import dataset
dataset = np.loadtxt('D:\GitHub\CS_4622_ML\Term Project\dataset.csv', 
                        delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 5, 8])

#split into features and target
X = dataset[:,0:5]
y = dataset[:,5]

#scale x data
X_scaled = scaler.fit_transform(X)

#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#create classifier
classifier = SVC(kernel='linear', decision_function_shape='ovr', C=0.6)
#fit classifier to training data
classifier.fit(X_train, y_train)

#asses the accuracy of the classifier
accuracy = classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)

print("\nSVM Classifier")
print("----------------------------------------------")
print(f"Accuracy (with colors): {(accuracy*100):.2f}%")
print(classification_report(y_test, y_pred))

#------------------------------------------------------
#  TESTING WITHOUT COLORS
#------------------------------------------------------

#split into features and target
X_NC = dataset[:,0:4]
y_NC = dataset[:,5]

#scale x data
X_NC_scaled = scaler.fit_transform(X_NC)

#split into train and test sets
X_NC_train = X_train[:,0:4]
X_NC_test= X_test[:,0:4]
y_NC_train, y_NC_test = y_train, y_test

#create classifier
classifier_2 = SVC(kernel='linear', decision_function_shape='ovr', C=0.6)
#fit classifier to training data
classifier_2.fit(X_NC_train, y_NC_train)

#asses the accuracy of the classifier
accuracy_NC = classifier_2.score(X_NC_test, y_NC_test)
y_pred_NC = classifier_2.predict(X_NC_test)

print("\nSVM Classifier")
print("----------------------------------------------")
print(f"Accuracy (without colors): {(accuracy_NC*100):.2f}%")
print(classification_report(y_NC_test, y_pred_NC))

# class_0 = X_NC[y_NC==0]
# class_1 = X_NC[y_NC==1]
# class_2 = X_NC[y_NC==2]
class_3 = X_NC[y_NC==3]
class_4 = X_NC[y_NC==4]
# class_5 = X_NC[y_NC==5]

print(f"Similarity between Class 3 and Class 4: {cosine_similarity(class_3, class_4)[0][0]}")