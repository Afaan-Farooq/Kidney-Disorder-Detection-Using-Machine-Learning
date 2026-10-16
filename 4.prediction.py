#This is the fourth and last step

# SVM Classification
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

#load dataset
url = "final_dataset.csv"
dataframe = pd.read_csv(url)
dataframe.head()
dataframe.describe()
array = dataframe.values
# Splitting the dataset into training and testing sets.
x = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
x_train, x_test, y_train, y_true = train_test_split(x, y, random_state = 7, test_size = 0.7)

# Creating the SVM model.
model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")

# define one new instance
Xnew = [[3,48,70,1.005,4,0,1,0,1,0,117,56,3.8,111,2.5,11.2,32,6700,3.9,1,0,0,0,1,1]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))

# define one new instance
Xnew = [[293,50,1.02,0,0,1,1,0,0,92,19,1.2,150,4.8,14.9,48,4700,5.4,0,0,0,1,0,0,0]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))