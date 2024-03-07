import numpy as np
import jax.numpy as jnp
import os
import copy

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier

from utils import *

class MetricOpt:
    def __init__(self, dim=None, lb=-0.2, ub=0.2):
        dataset = datasets.fetch_covtype()
        X = dataset.data
        y = dataset.target
        
        ckpt = 'model/MetricOpt.npy'
        hidden_layer_sizes = (30, 14)
        
        self.model = MLPClassifier
        
        self.dim = dim
        self.lb = lb * jnp.ones(self.dim)
        self.ub = ub * jnp.ones(self.dim)

        scaler = StandardScaler()  
        scaler.fit(X) 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.5, random_state=0)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        if os.path.exists(ckpt):
            self.clf = self.model(solver='lbfgs', alpha=1e-5, 
                                  hidden_layer_sizes=hidden_layer_sizes, 
                                  max_iter=10,random_state=1)
            self.clf.fit(self.X_train, self.y_train)
            with open(ckpt, 'rb') as f:
                self.params = np.load(f)
            self._reset_params(SCALE * np.ones(dim) / 2)
        else:
            self.clf = self.model(solver='lbfgs', alpha=1e-5, 
                                  hidden_layer_sizes=hidden_layer_sizes, 
                                  max_iter=300,random_state=1)
            self.clf.fit(self.X_train, self.y_train)
            
        self.score_func = lambda x,y: precision_score(y, self.clf.predict(x), average='macro')
        # self.score_func = lambda x,y: recall_score(y, self.clf.predict(x), average='macro')
        # self.score_func = lambda x,y: f1_score(y, self.clf.predict(x), average='macro')
        # self.score_func = lambda x,y: jaccard_score(y, self.clf.predict(x), average='macro')
            
        score = self.score_func(self.X_train, self.y_train)
        print('init error', 1 - score)
        
        self.params = []
        for w in self.clf.coefs_ + self.clf.intercepts_:
            self.params += [w.reshape(-1)]
        self.params = np.concatenate(self.params)
        print('len of params', len(self.params))
            
        with open(ckpt, 'wb') as f:
            np.save(f, self.params)
            
        
    def _reset_params(self, x):
        x = unnormalize(x, self.lb, self.ub, SCALE)
        x = np.array(x)
        params = copy.deepcopy(self.params)
        params[-self.dim:] += x
        self.clf._unpack(params)
        

    def __call__(self, x, test=False):
        self._reset_params(x)
        if test:
            score = self.score_func(self.X_test, self.y_test)
        else:
            score = self.score_func(self.X_train, self.y_train)
        return 1 - score
    
    
        