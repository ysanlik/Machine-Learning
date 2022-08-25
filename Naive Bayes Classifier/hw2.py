#!/usr/bin/env python
# coding: utf-8
#Yiğithan Şanlık 64117 HW2

# In[1]:


# importing the liabraries
import numpy as np
import pandas as pd

# defining fuction for safelog
def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


# loading the dataset and labels
data = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
labels = (np.genfromtxt("hw02_data_set_labels.csv", delimiter=",")).astype(int)


# In[3]:


# Find number of classes
classes = np.unique(labels)
print("classes =",classes)
K = len(classes)

#Shape of dataset
print("The dataset contains total {} samples".format(data.shape[0]))


# In[4]:


# Splitting the dataset into train and test 25 points from each class in train and 14 in test.
X_train=[]
X_test=[]
y_train=[]
y_test=[]
for i in range(K):
    for j in range(25):
        X_train.append(data[j+ 39*i])
        y_train.append(labels[j+ 39*i])
    for k in range(14):
        X_test.append(data[25 +k+ 39*i])
        y_test.append(labels[25 +k+ 39*i])
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


# In[5]:


# finding the class priors
class_priors = [np.mean(y_train == (c)) for c in range(1,K+1)]
#finding the pcd
pcd = [np.mean(X_train[y_train == (c)],axis=0) for c in range(1,K+1)]
pcd = np.reshape(pcd,(5,320)) # the dataset contains 5 calsses and 320 pixels


# In[6]:


print("PCD  = ")
print(pcd)


# In[7]:


print("Class Priors  = ")
print(class_priors)


# In[8]:


# using naive bayes for the the train scores
train_score = []
for i in range(X_train.shape[0]):
    train_score.append([np.sum(X_train[i] * safelog(pcd[c-1] ) + (1-X_train[i]) * safelog(1-pcd[c-1]))  + safelog(class_priors[c-1]) for c in range(1,K+1)])
train_score= np.array(train_score)


train_predicted = np.argmax(train_score, axis = 1)+1

# confusion matrix
confusion_matrix = pd.crosstab(train_predicted, y_train+1, rownames = ['y_pred'], colnames = ['y_truth'])
print("Confusion matrix for train data")
print(confusion_matrix)


# In[9]:


# using naive bayes for the the test scores
test_score = []
for i in range(X_test.shape[0]):
    test_score.append([np.sum(X_test[i] * safelog(pcd[c-1] ) + (1-X_test[i]) * safelog(1-pcd[c-1]))  + safelog(class_priors[c-1]) for c in range(1,K+1)])
test_score= np.array(test_score)


test_predicted = np.argmax(test_score, axis = 1)+1

# confusion matrix
confusion_matrix = pd.crosstab(test_predicted, y_test+1, rownames = ['y_pred'], colnames = ['y_truth'])
print("Confusion matrix for test data")
print(confusion_matrix)



