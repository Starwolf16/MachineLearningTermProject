import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# import dataset
data = np.loadtxt('Term Project/dataset.csv',
                        delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 5, 8])

# split data into features and target
x = data[:,0:4]
y = data[:,5]

# separate data for x with color feature
x_c = data[:,0:5]


# scale feature data
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_c = scaler.fit_transform(x_c)


# function for KNN that lets you set the number of neighbors, the random state, and test size
# returns the accuracy of the test
def knn(X, Y, neighbors=1, randomstate=None, testsize=0.3):
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=testsize, random_state=randomstate)
    k = KNeighborsClassifier(n_neighbors=neighbors)
    k.fit(x_train, y_train)
    y_pred = k.predict(x_test)

    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


# function for running multiple tests and storing their average and minimum accuracies in a list
def run(X, Y):
    accuracies = []
    for i in range(1, 12, 2):
        acc = []
        for j in range(100):
            acc.append(knn(X, Y, i, j))
        accuracies.append([np.mean(acc), np.min(acc)])
    return accuracies

# lists for the accuracies of the set without colors and the set with colors
accuracy = run(x, y)
accuracy_c = run(x_c, y)

# printing results
print(f'\n{" ":13}Without colors\t\t{" ":15}With colors')
print(f'{" ":10}Avg{" ":13}Min\t\t{" ":18}Avg{" ":13}Min')
for i in range(len(accuracy)):
    print(f'n={(i * 2) + 1:3}\t{accuracy[i][0]:<10.5}\t{accuracy[i][1]:<10.5}\t\t'
          f'n={(i * 2) + 1:3}\t{accuracy_c[i][0]:<10.5}\t{accuracy_c[i][1]:<10.5}')
    
print("")
