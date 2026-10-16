#This is the third step

# Logistic Regression Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
array = array.astype('int64')
X = array[:,0:25]
Y = array[:,25]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Logistic regression :")
print(results.mean())

# LDA Classification
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:,0:25]
Y = array[:,25]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LinearDiscriminantAnalysis()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("LDA :")
print(results.mean())

# KNN Classification
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:,0:25]
Y = array[:,25]
random_state = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("KNN :")
print(results.mean())

# Gaussian Naive Bayes Classification
import pandas
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:,0:25]
Y = array[:,25]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Guassian Naive Bayes Classification :")
print(results.mean())

# CART Classification
import pandas
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:,0:25]
Y = array[:,25]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = DecisionTreeClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(" Bagged cart or just decision trees :")
print(results.mean())
