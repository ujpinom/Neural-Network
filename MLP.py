import numpy as np
import sys

class MLP():

    def __init__(self,l_units=30,l2=0,epochs=100,eta=0.001,shuffle=True,mini=1,seed=None):
        self.l_units=l_units
        self.l2=l2
        self.epochs=epochs
        self.eta=eta
        self.shuffle=shuffle
        self.mini=mini
        self.random=np.random.RandomState(seed)

    def oneHot(self,y,numero_clases):

        onehot=np.zeros((y.shape[0],numero_clases))

        for index,value in enumerate(y.astype(int)):
            onehot[value,index]=1

        return onehot

    def funcion_sigmoid_(self,z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def forward_propagation_(self,X):

        z_h=np.dot(X,self.w_h)+self.b_h
        a_h=self.funcion_sigmoid_(z_h)

        z_out=np.dot(a_h,self.w_out)+ self.b_out

        a_out=self.funcion_sigmoid_(z_out)

        return z_h,a_h,z_out,a_out

    def _compute_cost(self, y_enc, output):

        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self,X):

        z_h, a_h, z_out, a_out=self.forward_propagation_(X)
        y_pred=np.nanargmax(z_out,axis=1)

        return y_pred

    def fit(self,X_train,y_train,X_test,y_test):

        numero_clases=len(np.unique(y_train))
        numero_caracteristicas=X_train.shape[1]

        self.b_h = np.zeros(self.l_units)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,size=(numero_caracteristicas,self.l_units))
        self.b_out = np.zeros(numero_clases)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                    size=(self.l_units,
                                          numero_clases))

        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self.oneHot(y_train, numero_caracteristicas)

        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.mini + 1, self.mini):
                batch_idx = indices[start_idx:start_idx + self.mini]
                z_h, a_h, z_out, a_out = self.forward_propagation_(X_train[batch_idx])

                sigma_out = a_out - y_train_enc[batch_idx]

                sigmoid_derivative_h = a_h * (1. - a_h)

                sigma_h = (np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h)

                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out


            z_h, a_h, z_out, a_out = self.forward_propagation_(X_train)
            cost = self._compute_cost(y_enc=y_train_enc,output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid ==y_valid_pred)).astype(np.float) / X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% '
                             %
                             (epoch_strlen, i + 1, self.epochs,
                              cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

        

