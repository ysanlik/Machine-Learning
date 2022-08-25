#!/usr/bin/env python
# coding: utf-8
# Yigithan Sanlik 64117

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_set = np.genfromtxt("hw04_data_set_train(1).csv", delimiter = ",")
x_train = train_set[:,0]
y_train = train_set[:,1]



test_set = np.genfromtxt("hw04_data_set_test(1).csv", delimiter = ",")
x_test = test_set[:,0]
y_test = test_set[:,1]


train_size = x_train.shape[0]
test_size = x_test.shape[0]


# In[3]:


origin = 0
bin_width = 0.1
minimum_value = origin
maximum_value = 60 # from hw pdf 


# In[4]:


left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

p_hat = np.asarray([np.sum(((left_borders[i] < x_train) & (x_train <= right_borders[i])) * y_train)/np.sum((left_borders[i] < x_train) & (x_train <= right_borders[i])) for i in range(len(left_borders))])

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
#plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()


# In[5]:


#left_borders = np.arange(minimum_value, maximum_value, bin_width)
#right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

p_hat = np.asarray([np.sum(((left_borders[i] < x_train) & (x_train <= right_borders[i])) * y_train)/np.sum((left_borders[i] < x_train) & (x_train <= right_borders[i])) for i in range(len(left_borders))])

#plt.figure(figsize = (10, 6))
#plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()


# In[6]:


def rmse(methodName,p_hat,bin_width):
    rmse = 0
    if methodName=="Regressogram":
        for i in range(test_size):
            for b in range(len(right_borders)):
                if (left_borders[b] < x_test[i]) and (x_test[i] <= right_borders[b]):
                    rmse += (y_test[i] - p_hat[b])**2
        rmse = math.sqrt(rmse/test_size) 
    else: 
        for i in range(test_size):
            index=-1
            minVal=10000000
            for x in range(len(data_interval)):
                value=abs(x_test[i]-data_interval[x])
                if value<minVal:
                    minVal=value
                    index=x
            rmse += (y_test[i] - p_hat[index])**2
        rmse = math.sqrt(rmse/test_size)
        
    print(methodName,"=> RMSE is", rmse, "when h is", bin_width)


# In[7]:


rmse("Regressogram",p_hat,bin_width)


# In[8]:


data_interval = np.linspace(minimum_value, maximum_value, int((maximum_value-minimum_value)*100)+1)

#p_hat=np.asarray([np.sum(((((x_train - x)/bin_width) <= 0.5) & (((x_train - x)/bin_width) >= -0.5))*y_train)/np.sum((((x_train - x)/bin_width) <= 0.5) & (((x_train - x)/bin_width) >= -0.5)) for x in data_interval])
p_hat=np.asarray([np.sum((np.abs((x_train - x)/bin_width) <= 0.5)*y_train)/
                  np.sum(np.abs((x_train - x)/bin_width) <= 0.5)for x in data_interval])


plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
plt.plot(data_interval, p_hat, "k-")
plt.show()


# In[9]:


plt.figure(figsize = (10, 6))
plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
plt.plot(data_interval, p_hat, "k-")
plt.show()


# In[10]:


rmse("Running Mean Smoother",p_hat,bin_width)


# In[11]:


bin_width = 0.02


# In[12]:


def KernelFuncNumerator(x):
    return np.sum((1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2))*y_train)


# In[13]:


def KernelFunc(x):
    return np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2))


# In[14]:


p_hat = np.asarray([KernelFuncNumerator(x)/KernelFunc(x) for x in data_interval])

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
#plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
plt.plot(data_interval, p_hat, "k-")
plt.show()


# In[15]:



plt.figure(figsize = (10, 6))
#plt.plot(x_train, y_train, "b.", markersize = 10, label = "training")
plt.plot(x_test, y_test, "r.", markersize = 10, label = "test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc = 'upper left')
plt.plot(data_interval, p_hat, "k-")
plt.show()


# In[16]:


rmse("Kernel Smoother",p_hat,bin_width)


# In[ ]:





# In[ ]:




