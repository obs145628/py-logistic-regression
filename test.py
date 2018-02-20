from sklearn.linear_model import LogisticRegression
import dataset_mnist
import logreg

'''
Apply logistic regression with sklearn
'''
def train_sclearn(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(fit_intercept=True, C = 1e15)
    print('Training with sklearn:')
    clf.fit(X_train, y_train)
    print('Train set accuracy: ' + str(clf.score(X_train, y_train)))
    print('Test  set accuracy: ' + str(clf.score(X_test, y_test)))


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = dataset_mnist.load_mnist_bin()
    
    #train_sclearn(X_train, y_train, X_test, y_test)   
    logreg.train(X_train, y_train, X_test, y_test, 30, 0.001, use_intercept = True, batch_size = -1)
