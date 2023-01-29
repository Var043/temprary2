# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  importing dataset 

data_set=pd.read_csv("Iris.csv")
print(data_set.head(10))

# extracting independent and dependent variables

x=data_set.iloc[:,[1,2,3,4]].values  #independent: sepallength,width & petallength,width
y=data_set.iloc[:,5].values    #dependent: species
from sklearn.preprocessing import LabelEncoder #convert textlabels to integers
le=LabelEncoder()
y=le.fit_transform(y)

# print(list(y))   after conversion text lebels to int, below is output
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
# 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 

# now we split the dataset into a training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# now we do feature scaling because values are lie in diff ranges

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)


# # finally we are trianing our logistic regression model

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=0)
classifier.fit(x_train,y_train)

# after training model its time to use it to do predictions on testing data.

y_pred=classifier.predict(x_test)


# confusion_matrix: use to check the model performance...

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix : \n",cm)

# getting accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(y_test,y_pred))


# import pickle
# # saving the model a pickle file
# pickle.dump(DecisionTreeClassifier,open('DT_model.pkl','wb'))

# # loading the model to disk
# pickle.dump(DecisionTreeClassifier,'DT_model.pkl','rb')


# visualising the dataset

iris=pd.read_csv("Iris.csv")
# print(iris.head())

fig,ax=plt.subplots()
ax.set_title(' Iris Dataset ')
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
colors={'Iris-setosa':'r','Iris-versicolor':'b','Iris-virginica':'g'}
for i in range(len(iris['SepalLengthCm'])):
    ax.scatter(iris['SepalLengthCm'][i],iris['SepalWidthCm'][i],
    color=colors[iris['Species'][i]])
ax.legend()
plt.show()


# Visualising using seaborn 
# import seaborn as sns

# sns.set_style("whitegrid")
# sns.FacetGrid(iris,hue="Species",
#                 height=6).map(plt.scatter,
#                 'SepalLengthCm',
#                 'SepalWidthCm').add_legend()
# plt.show()