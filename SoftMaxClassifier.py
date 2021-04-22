
import numpy as np

class SFT():



    def one_hot_encoder_(self,y):
        m=len(y)
        onehot=np.zeros((m,y.max() + 1))
        onehot[np.arange(m),y]=1
        return onehot

    def __init__(self,eta=0.1,alpha=0.1,n_iter=5000,random_state=42):

        self.eta=eta
        self.alpha=alpha
        self.n_iter=n_iter
        self.random=random_state



    def fit(self,X,y):

        np.random.seed(self.random)
        self.X_bias=np.c_[np.ones([len(X),1]),X]
        m=len(self.X_bias)
        n_outputs = len(np.unique(y))
        n_inputs = self.X_bias.shape[1]
        self.Theta_=np.random.randn(n_inputs,n_outputs)

        for iteracion in range(self.n_iter):

            logi=self.X_bias.dot(self.Theta_)

            probabilidades=self.softmax_(logi)

            Y_train= self.one_hot_encoder_(y)

            error=probabilidades-Y_train

            gradiente = 1/m*self.X_bias.T.dot(error)+np.r_[np.zeros([1,n_outputs]),self.alpha*self.Theta_[1:]]

            self.Theta_=self.Theta_-self.eta*gradiente

        return self

    def predict(self,X):
        X_bias = np.c_[np.ones([len(X), 1]), X]
        propabilidades=self.softmax_(X_bias.dot(self.Theta_))
        return np.argmax(propabilidades,axis=1)

    def softmax_(self,logit):

        num=np.exp(logit)
        den=np.sum(num,axis=1,keepdims=True)
        return num/den







