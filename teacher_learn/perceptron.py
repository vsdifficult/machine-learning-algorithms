from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from sklearn.metrics import confusion_matrix 
from scipy.stats import randint 

import numpy as np, seaborn as sns, matplotlib.pyplot as plt 

class Perceptron: 

    def __init__(self, num_epochs, dim):
        self.num_epochs = num_epochs 
        self.theta0 = 0 
        self.theta = np.zeros(dim) 

    def fit(self, X_train, y_train): 
        n = X_train.shape[0]
        dim = X_train.shape[1]

        k = 1 
        for epoch in range(self.num_epochs): 
            for i in range(n): 
                idx = randint.rvs(0, n-1, size=1)[0]
                if (y_train[idx] * (np.dot(self.theta, X_train[idx, :]) + self.theta0) <= 0): 
                    eta = pow(k+1, -1)
                    k += 1 
                    self.theta = self.theta + eta * y_train[idx] * X_train[idx, :]
                    self.theta0 = self.theta0 + eta * y_train[idx] 

            print("Epoch: ", epoch)
            print("Theta: ", self.theta)
            print("Theta0: ", self.theta0) 


    def predict(self, X_test): 
        n = X_test.shape[0]
        dim = X_test.shape[1] 
        y_pred = np.zeros(n)
        for idx in range(n): 
            y_pred[idx] = np.sign(np.dot(self.theta, X_test[idx, :]) + self.theta0) 
        return y_pred 

if __name__ == "__main__": 
    iris = load_iris()
    X = iris.data[:100, :]
    y = iris.target[:100] - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    clf = Perceptron(num_epochs=5, dim=X.shape[-1]) 
    clf.fit(X_train, y_train) 

    y_pred = clf.predict(X_test) 

    cmt = confusion_matrix(y_test, y_pred) 
    acc = np.trace(cmt)/np.sum(np.sum(cmt)) 

    print("Perceptron accuracy: ", acc) 

    plt.figure() 
    sns.heatmap(cmt, annot=True, fmt="d")
    plt.title("Error Matrix"); plt.xlabel("Predict") ; plt.ylabel("Factional") 
    plt.savefig("./teacher_learn/perceptron_acc.png") 
    plt.show()
