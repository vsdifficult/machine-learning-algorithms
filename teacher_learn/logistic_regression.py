import numpy as np, matplotlib.pyplot as plt 

def generate_data(): 
    n = 1000 
    mu1 = np.array([1, 1])
    mu2 = np.array([-1, -1]) 

    pik = np.array([0.4, 0.6]) 
    
    X = np.zeros((n, 2))
    y = np.zeros((n, 1)) 

    for i in range(1,n): 
        u = np.random.rand()
        idx = np.where(u < np.cumsum(pik))[0] 
        if (len(idx) == 1): 
            X[i, :] = np.random.randn(1, 2) + mu1
            y[i] = 1 

        else: 
            X[i, :] = np.random.randn(1, 2) + mu2
            y[i] = -1 

    return X, y 

class sgdlr: 

    def __init__(self):
        self.num_iter = 100 
        self.lmbda = 1e-9

        self.tau0 = 10
        self.kappa = 1 
        self.eta = np.zeros(self.num_iter) 

        self.batch_size = 200 
        self.eps = np.finfo(float).eps


    def fit(self, X, y): 

        theta = np.random.randn(X.shape[1], 1) 

        for i in range(self.num_iter):
            self.eta[i] = (self.tau0 + i) ** (-self.kappa) 
        
        batch_data, batch_labels = self.make_batches(X, y, self.batch_size) 
        num_batches = batch_data.shape[0] 
        num_updates = 0 

        J_hist = np.zeros((self.num_iter * num_batches, 1)) 
        t_hist = np.zeros((self.num_iter * num_batches, 1))

        for itr in range(self.num_iter): 
            for b in range(num_batches): 
                Xb = batch_data[b]
                yb = batch_labels[b] 

                J_cost, J_grad = self.lr_objective(theta, Xb, yb, self.lmbda) 
                theta = theta - self.eta[itr] * (num_batches * J_grad) 

                J_hist[num_updates] = J_cost
                t_hist[num_updates] = np.linalg.norm(theta, 2)

                num_updates = num_updates + 1 
            print("iteration %d, cost %f" %(itr, J_cost)) 
        
        y_pred = 2*(self.sigmoid(X.dot(theta)) > 0.5) - 1 
        y_err = np.size(np.where(y_pred - y)[0])/float(y.shape[0]) 
        print("classification error:", y_err) 
        self.generate_plots(X, J_hist, t_hist, theta)
        return theta
    
    def make_batches(self, X, y, batch_size): 
        n = X.shape[0]
        d = X.shape[1] 

        num_batches = int(np.ceil(n/batch_size)) 

        groups = np.tile(range(num_batches), batch_size)
        batch_data = np.zeros((num_batches, batch_size,d)) 
        batch_labels = np.zeros((num_batches, batch_size, 1)) 

        for i in range(num_batches): 
            batch_data[i, :, :] = X[groups==i, :] 
            batch_labels[i, :] = y[groups==i] 
        return batch_data, batch_labels 
    
    def lr_objective(self, theta, X, y, lmbda): 
        n = y.shape[0] 
        y01 = (y+1)/2.0 

        mu = self.sigmoid(X.dot(theta)) 
        mu = np.maximum(mu, self.eps) 
        mu = np.minimum(mu, 1-self.eps) 

        data_cost = -np.sum(y01 * np.log(mu) + (1-y01) * np.log(1-mu)) / n 
        reg_cost = lmbda * np.sum(theta * theta) 
        cost = data_cost + reg_cost

        grad = X.T.dot(mu-y01) / n + 2*lmbda*theta 

        return cost, grad

    def sigmoid(self, a): 
        return 1 / (1+np.exp(-a)) 
    
    def generate_plots(self, X, J_hist, t_hist, theta): 
        plt.figure() 
        plt.plot(J_hist) 
        plt.title("Logistic Regression") 
        plt.xlabel("Iterations")
        plt.ylabel("Loss") 
        plt.show() 

        plt.figure() 
        plt.plot(t_hist) 
        plt.title("LR theta 12 norm")
        plt.xlabel("Iterations")
        plt.ylabel("Theta 12 norm") 
        plt.show() 

        plt.figure()
        plt.plot(self.eta) 
        plt.title("LR learning rate") 
        plt.xlabel("Iterations")
        plt.ylabel('learning rate')
        plt.show()

        plt.figure()
        x1 = np.linspace(np.min(X[:,0]) - 1, np.max(X[:, 0])+1,10) 
        plt.scatter(X[:,0], X[:, 1])
        plt.plot(x1, -(theta[0]/theta[1])*x1)
        plt.title("Decision Boundary") 
        plt.grid(True) 
        plt.xlabel("X1")
        plt.ylabel("X2") 
        plt.show() 

if __name__ == "__main__": 

    X, y = generate_data() 
    sgd = sgdlr() 
    theta = sgd.fit(X, y)