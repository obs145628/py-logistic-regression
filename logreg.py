'''
Logistic Regression implementation

more informations:
- http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/
- https://beckernick.github.io/logistic-regression-from-scratch/
'''



import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
Compute log likelihood (cost function) on data set of size n
@param x - matrix (n * nb_features) matrix of features for each example
@param y - vector (n) vector of labels, on for each example
@param w - vector (nb_features) vector of weights
'''
def log_likelihood(x, y, w):
    z = np.dot(x, w)
    return - np.sum(y*z - np.log(1 + np.exp(z)))



'''
Apply stochastic gradient descent on the whole training set of size n
@param x - matrix (n * nb_features) matrix of features for each example
@param y - vector (n) vector of labels, on for each example
@param w - vector (nb_features) vector of weights
@param lr - learning rate
@return  vector(nb_features) updated weights
'''
def sgd(x, y, w, lr):

    y_hat = sigmoid(np.dot(x, w))
    dW = np.dot(x.T, y_hat - y)
    w = w - lr * dW
    return w

'''
Apply stochastic_gradient_descent on the whole training set of size n, using mini_batches
@param x - matrix (n * nb_features) matrix of features for each example
@param y - vector (n) vector of labels, on for each example
@param w - vector (nb_features) vector of weights
@param lr - learning rate
@param batch_size - size of each batch
@return  vector(nb_features) updated weights
'''
def sgd_mini_batch(X, y, w, lr, batch_size):

    n = X.shape[0]

    for k in range(0, n, batch_size):
        X_batch = X[k:k + batch_size]
        y_batch = y[k:k + batch_size]
        m = len(X_batch)

        y_hat = sigmoid(np.dot(X_batch, w))
        dW = np.dot(X_batch.T, y_hat - y_batch)
        w = w - (lr / m) * dW
    return w


def evaluate(X, y, w):

    y_hat = np.round(sigmoid(np.dot(X, w)))
    succ = int(np.sum(y_hat == y))
    total = X.shape[0]
    perc = succ * 100 / total

    print('Cost: ' + str(log_likelihood(X, y, w)))
    print('Results: {} / {} ({}%)'.format(succ, total, perc))


'''
@param X_train - matrix (n * nb_features) matrix of features (training set)
@param y_train - vector (n) vector of labels (training set)
@param x_test - matrix (n * nb_features) matrix of features (test set)
@param y_test - vector (n) vector of labels (test set)
@param epochs - number of epochs of learning
@param use_intercept - if true, add a bias to the weights
@param batch_size - size of batch size for gradient descent, if -1, gradient descent is on the full dataset

Run the training for several epochs.
After each epochs, the wieghts are tested on the training test and the test set

'''
def train(X_train, y_train, X_test, y_test, epochs, lr, use_intercept = False, batch_size = -1):

    if use_intercept:
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    w = np.zeros(X_train.shape[1])

    #Training
    for i in range(1, epochs + 1):
        print('Epoch :' + str(i))
        if batch_size == -1:
            w = sgd(X_train, y_train, w, lr)
        else:
            w = sgd_mini_batch(X_train, y_train, w, lr, batch_size)
        print('Train:')
        evaluate(X_train, y_train, w)
        print('Test:')
        evaluate(X_test, y_test, w)

    return w




if __name__ == '__main__':
    X_train, y_train, X_test, y_test = dataset_mnist.load_mnist_bin()
    
    
    #train_sclearn(X_train, y_train, X_test, y_test)   
    train(X_train, y_train, X_test, y_test, 30, 0.001, use_intercept = True, batch_size = -1)
