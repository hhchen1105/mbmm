#from scipy.special import psi
from numpy import log
import numpy as np
from scipy.stats import beta
from scipy.special import gamma
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.preprocessing import normalize
#from scipy.optimize import root_scalar

class MBMM:
    def __init__(self, C, n_runs, param):
        self.C = C # number of Guassians/clusters
        self.n_runs = n_runs
        self.param = param
        self.pi = np.array([1./self.C for i in range(self.C)])

    def get_params(self):
        return [self.param, self.pi]
    
    def get_pi(self):
        return self.pi

    def fit(self, X):
        '''
        Parameters:
        -----------
        X: (N x d), data 
        '''
        N = X.shape[0]
        d = X.shape[1]
        
              
        #old_loss = float('-inf')
        try:
            for run in range(self.n_runs):         
                self.e_step(X)
                self.m_step(X)
                         
                
                #loss = self.compute_loss_function(X) 
                #if abs(old_loss-loss) < 1e-8:          
                #    break
            print('fit done!')  
               
        except Exception as e:
            print(e)
            
    def e_step(self, X):
        #X: (N x d)
        #self.gamma: (N x C)
        #self.alphas: (C x d)
        N = X.shape[0]      
        D = X.shape[1]
        self.gamma = np.zeros((N, self.C))
        
 
        for c in range(self.C): 
            self.gamma[:,c] = self.pi[c]*self.mpdf(X,self.param[c,:])
           
        for i in range(N):
            self.gamma[i,:] /= (np.sum(self.gamma[i,:]))
        
    def mpdf(self, X, param):
    

        N = X.shape[0]
        m = X.shape[1]
 
        top = 1.0
        for j in range(m):
            top *= pow(X[:,j],param[j]-1)/(pow(1-X[:,j],param[j]+1))
            
        
        b_func = 1.0
        for j in range(m+1):
            b_func *= gamma(param[j])
            
        #avoid overflow
        if np.sum(param)>=170:
            b_func = 1e-8
        else:
            b_func = b_func / gamma(np.sum(param))
        
        down = 1.0
        for j in range(m):
            down += X[:,j]/(1-X[:,j])
     
 
        down = pow(down, np.sum(param))
        
       
        return top/(b_func*down)
        
    
    def m_step(self, X):
        
                
        N = X.shape[0]      
        d = X.shape[1] 
               
        #param (C*d)
        param_num = self.C*(d+1) # total parameters num
        x_guess = np.array([])
        lower = np.array([])
        upper = np.array([])
        
        #initialize optimizatin parameters and boundary
        for i in range(param_num):
            x_guess = np.append(x_guess,self.param[i//(d+1)][i%(d+1)])
            lower = np.append(lower, 1e-8) #lower
            upper = np.append(upper, 50.) #upper
        
        def new_loglikeli(p):        

            total = 0
            for i in range(N):
                temp = 0
                for c in range(self.C):
                    temp += self.pi[c]*self.mpdf(np.array([X[i,:]]),p[c*(d+1):(c+1)*(d+1)])
            
                total += log(temp)
                #print(self.pi[0]*self.pdf(X[i,0],X[i,1],p[0],p[1],p[2])+self.pi[1]*self.pdf(X[i,0],X[i,1],p[3],p[4],p[5]))

            return -total        
        
 
        bounds = Bounds(lower, upper)
        res = minimize(new_loglikeli, x_guess, method='SLSQP',options={'ftol': 1e-9}, bounds=bounds)
        
        for i in range(param_num):
            self.param[i//(d+1)][i%(d+1)] = res.x[i]
     

        #pi (C)
        for c in range(self.C):
            self.pi[c] = np.sum(self.gamma[:,c]) / N
    
    def predict(self, X):      
        N = X.shape[0]  
        labels = np.zeros((N, self.C))
 
        for c in range(self.C):
            labels[:,c] = self.pi[c]*self.mpdf(X,self.param[c,:])
            
        labels  = labels.argmax(1)
        
        return labels

    def predict_proba(self, X):
        N = X.shape[0]  
        labels = np.zeros((N, self.C))
 
        for c in range(self.C):
            labels[:,c] = self.pi[c]*self.mpdf(X,self.param[c,:])
        scores = normalize(labels, axis=1, norm='l1')
            
        return scores

    def compute_loss_function(self, X):
        
        N = X.shape[0]  
        
        total = 0.0
        for i in range(N):
            temp = 0.0
            for c in range(self.C):
                #mpdf must be 2-dim
                temp += self.pi[c]*self.mpdf(np.array([X[i,:]]),self.param[c,:])
            total += log(temp)
     
        return total    
       
