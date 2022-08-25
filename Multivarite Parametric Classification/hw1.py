 #Yiğithan Şanlık 64117, HW1, ENGR421.
 #Run it in Jupyter Notebook.
 #There might be an issue on Part V if you run it in PyCharm.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(100)

# Defining the class sizes
class1_size = 105
class2_size = 145
class3_size = 135
class4_size = 115

# Defining class means
class1_mean = [+0.0, +4.5]
class2_mean = [-4.5, -1.0]
class3_mean = [+4.5, -1.0]
class4_mean = [+0.0, -4.0]

# Defining class covariences
class1_covariance = [[+3.2, +0.0], [+0.0, +1.2]]
class2_covariance = [[+1.2, +0.8], [+0.8, +1.2]]
class3_covariance = [[+1.2, -0.8], [-0.8, +1.2]]
class4_covariance = [[+1.2, +0.0], [+0.0, +3.2]]

# generate random samples
class1 = np.random.multivariate_normal(class1_mean, class1_covariance, class1_size)
class2 = np.random.multivariate_normal(class2_mean, class2_covariance, class2_size)
class3 = np.random.multivariate_normal(class3_mean, class3_covariance, class3_size)
class4 = np.random.multivariate_normal(class4_mean, class4_covariance, class4_size)

# stack them to crete the dataset
X = np.vstack((class1, class2, class3, class4))

# generating the  labels
y = np.concatenate(
    (np.repeat(1, class1_size), np.repeat(2, class2_size), np.repeat(3, class3_size), np.repeat(4, class4_size)))

# Saving the file as csv
np.savetxt("data.csv", np.hstack((X, y[:, None])), fmt="%f,%f,%d")

# Ploting the data
plt.figure(figsize=(10, 10))
plt.plot(class1[:, 0], class1[:, 1], color='green', marker='o', markersize=10, linestyle="")
plt.plot(class2[:, 0], class2[:, 1], color='red', marker='o', markersize=10, linestyle="")
plt.plot(class3[:, 0], class3[:, 1], color='blue', marker='o', markersize=10, linestyle="")
plt.plot(class4[:, 0], class4[:, 1], color='purple', marker='o', markersize=10, linestyle="")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Loading the dataset
data = np.genfromtxt("data.csv", delimiter=",")

# Seperate data from class labels
X = data[:, [0, 1]]
y_true = data[:, 2].astype(int)

# get number of classes and number of samples
K = np.max(y_true)
N = data.shape[0]

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_true - 1] = 1

# calculate sample means
sample_means = [np.mean(X[y == (c + 1)], axis=0) for c in range(K)]
print("Sample Means", sample_means)

# calculate covariances
sample_covariances = [np.cov(np.transpose(X[y == c + 1])) for c in range(K)]
print("Sample Covarience", sample_covariances)

# calculate prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)


def safelog(x):
    return (np.log(x + 1e-100))


# define the softmax_classifier function
def softmax_classifier(X, W, w0):
    scores = np.matmul(np.hstack((X, np.ones((N, 1)))), np.vstack((W, w0)))
    scores = np.exp(scores - np.repeat(np.amax(scores, axis=1, keepdims=True), K, axis=1))
    scores = scores / np.repeat(np.sum(scores, axis=1, keepdims=True), K, axis=1)
    return (scores)


def gradient_descent(X, i):
    Wc = [-0.5 * np.linalg.inv(sample_covariances[c]) for c in range(K)]
    wc = [np.matmul(np.linalg.inv(sample_covariances[c]), sample_means[c]) for c in range(K)]
    wc0 = [
        -0.5 * np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])),
                         sample_means[c])
        + -0.5 * np.log(np.linalg.det(sample_covariances[c]))
        + np.log(class_priors[c])
        for c in range(K)]
    result = np.matmul(np.matmul(np.transpose(X), Wc[i]), X) + np.matmul(np.transpose(wc[i]), X) + wc0[i]
    return result


def gradient_W(X, y_true, y_predicted):
    return (np.asarray(
        [-np.sum(np.repeat((Y_truth[:, c] - Y_predicted[:, c])[:, None], X.shape[1], axis=1) * X, axis=0) for c in
         range(K)]).transpose())


def gradient_w0(Y_truth, Y_predicted):
    return (-np.sum(Y_truth - Y_predicted, axis=0))


eta = 0.01
epsilon = 1e-3

W = np.random.uniform(low=-0.01, high=0.01, size=(X.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

# learn W and w0 using gradient descent
iteration = 1
objective_values = []
while 1:

    Y_predicted = softmax_classifier(X, W, w0)
    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(Y_predicted)))
    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break

    iteration = iteration + 1

# plot the loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "r--")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Creating the confusion matrix
y_predicted = np.argmax(Y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_true, rownames=['y_pred'], colnames=['y_true'])
print(confusion_matrix)

# evaluate function on a grid
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:, :, c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
D = discriminant_values[:, :, 3]

A[(A < B) & (A < C) & (A < D)] = np.nan
B[(B < A) & (B < C) & (B < D)] = np.nan
C[(C < A) & (C < B) & (C < D)] = np.nan
D[(D < A) & (D < B) & (D < C)] = np.nan
discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C
discriminant_values[:, :, 3] = D

# Plotting the decision boundary
plt.figure(figsize=(10, 10))
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 0], levels=0, cmap='hot')
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, cmap='cool')
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 3], levels=0, cmap='spring')
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 3] - discriminant_values[:, :, 2], levels=0, cmap='Wistia')
plt.plot(X[y_true == 1, 0], X[y_true == 1, 1], color='green', marker='o', markersize=10, linestyle="")
plt.plot(X[y_true == 2, 0], X[y_true == 2, 1], color='red', marker='o', markersize=10, linestyle="")
plt.plot(X[y_true == 3, 0], X[y_true == 3, 1], color='blue', marker='o', markersize=10, linestyle="")
plt.plot(X[y_true == 4, 0], X[y_true == 4, 1], color='purple', marker='o', markersize=10, linestyle="")
plt.plot(X[y_predicted != y_true, 0], X[y_predicted != y_true, 1], "ko", markersize=10, fillstyle="none")

plt.xlabel("x1")
plt.ylabel("x2")






