# In[1]:

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# loading the dataset
labels = pd.read_csv("hw03_data_set_labels.csv",header = None).to_numpy()
data = pd.read_csv("hw03_data_set_images.csv",header = None).to_numpy()


# In[3]:


# get number of classes and number of samples
classes = np.unique(labels)
K = len(classes)
N = data.shape[0]
#Shape of dataset
print("The dataset contains total {} samples and {} classes.".format(N,K))


# In[4]:


# splitting the dataset for training and testing
y_train = []
y_test = []
X_train = np.zeros((1,320))
X_test = np.zeros((1,320))


ytruth = np.concatenate((np.repeat(1, 25), np.repeat(2,25), np.repeat(3, 25),np.repeat(4, 25),np.repeat(5, 25)))
ytruthtest = np.concatenate((np.repeat(1, 14), np.repeat(2,14), np.repeat(3, 14),np.repeat(4, 14),np.repeat(5, 14)))

Y_truth = np.zeros((125, 5)).astype(int)
Y_truth[range(125), ytruth - 1] = 1

Y_truthtest = np.zeros((70, 5)).astype(int)
Y_truthtest[range(70), ytruthtest - 1] = 1

for i in range(K):
    y_train = np.append(y_train,labels[i*39:i*39+25, :])
    y_test = np.append(y_test, labels[i*39+25:i*39+39, :])
    X_train = np.vstack([X_train, data[i*39:i*39+25, :]])
    X_test = np.vstack([X_test, data[i*39+25:i*39+39, :]])

# deleting theexta axis
X_train = np.delete(X_train, 0, 0)
X_test= np.delete(X_test, 0, 0)


# In[5]:


# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# In[6]:


# define the gradient functions
def gradient_W(X, y_truth, y_predicted):
    return np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(5)]).transpose()


# In[7]:


def gradient_w0(Y_truth, Y_predicted):
    return -np.sum(Y_truth - Y_predicted, axis = 0)


# In[8]:


eta = 0.001
epsilon = 0.001

W = np.random.uniform(low = -0.01, high = 0.01, size = (X_train.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# In[9]:


iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(X_train, W, w0)
    objective_values.append(np.sum((Y_truth - Y_predicted)**2)*(0.5))
    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(X_train, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)
    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    iteration = iteration + 1
print("W = ",W)
print("W0 = ",w0)


# In[10]:


# Iteration vs Loss plot
plt.plot(range(1, iteration + 1), objective_values, "r-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[11]:


# printing the confusion matrix
print("*******************FOR TRAINING DATA***********************")
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, ytruth, rownames = ['y_pred'], colnames = ['y_true'])
print(confusion_matrix)


print("*******************FOR TESTING DATA***********************")
ypredstest = sigmoid(X_test,W,w0)
y = np.argmax(ypredstest, axis = 1) + 1
confusion_matrix = pd.crosstab(y, ytruthtest, rownames = ['y_pred'], colnames = ['y_true'])
print(confusion_matrix)


# In[ ]:




