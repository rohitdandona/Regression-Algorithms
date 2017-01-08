from __future__ import division  # floating point division
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import warnings

import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest
        
class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params={} ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class RidgeRegression(Regressor):
    """
    Ridge Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5], 'lambda': 0.1}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)+(self.params['features'] * np.identity(Xless.shape[1]))), Xless.T),ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class MPLinearRegression(Regressor):
    """
    Linear Regression with matching pursuit
    """
    def __init__( self, params={} ):
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(params)

    def get_feature_with_max_coeff(self, residual, Xtrain, selected_list, features):
        coefficient_dict = {}
        for feature in features:
            if feature not in selected_list:
                Xless = Xtrain[:, feature]
                #coeff = np.absolute(np.dot(Xless.T,residual))
                #coeff = np.absolute(pearsonr(Xless,residual))
                #print Xless
                #print residual.shape
                coeff = np.absolute(np.corrcoef(Xless,residual)[1,0])
                warnings.filterwarnings("ignore")
                if not np.isnan(coeff):
                    coefficient_dict[coeff]=feature
        if coefficient_dict:
            return coefficient_dict[max(coefficient_dict)], max(coefficient_dict)
        else:
            return None, None

    def learn(self, Xtrain, ytrain):
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class StochasticGradientDescent(Regressor):
    """
    Linear Regression with Stochastic Gradient Descent
    """

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'epochs': 5, 'alpha': 0.01}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        #iterative_weights = np.random.random_sample(size=Xtrain.shape[1])
        iterative_weights = np.zeros(Xtrain.shape[1])
        #iterative_weights = np.random.randint(low = 50,high=100,size=Xtrain.shape[1])
        error_list = []
        for i in range(self.params['epochs']):
            dataset = np.insert(Xtrain, Xtrain.shape[1], ytrain, axis=1)
            np.random.shuffle(dataset)
            alpha = self.params['alpha']
            for t in range (Xtrain.shape[1]):
                tempy = dataset[:, Xtrain.shape[1]]
                tempx = np.delete(dataset, Xtrain.shape[1], axis=1)
                Xtrain_t = tempx[t,:]
                ytrain_t = tempy[t]

                #if t!=0:
                    #alpha = float(self.params['alpha'])/float(math.sqrt(t))

                #err_t = alpha * np.dot(np.subtract(np.dot(Xtrain_t.T, iterative_weights),ytrain_t),Xtrain_t)
                err_t = alpha * (np.dot((np.dot(Xtrain_t.T, iterative_weights) - ytrain_t),Xtrain_t))
                iterative_weights = iterative_weights - err_t

            self.weights = iterative_weights

            predictions = self.predict(Xtrain)
            err_w = (np.square(np.linalg.norm(np.subtract(predictions, ytrain)))) / ytrain.shape[0]
            error_list.append(err_w)

        #print error_list
        plt.plot(error_list)
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        #plt.show()
        # = plt.figure()
        plt.savefig('foo.png')

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class BatchGradientDescent(Regressor):
    """
    Linear Regression with Batch Gradient Descent
    """

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'alpha': 1, 'err':100000000, 'tolerance':0.0001}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        #iterative_weights = np.random.randint(low=50, high=100, size=Xtrain.shape[1])
        iterative_weights = np.zeros(Xtrain.shape[1])
        #iterative_weights = np.random.rand(Xtrain.shape[1])
        self.weights = iterative_weights

        predictions = self.predict(Xtrain)
        err_w = (np.square(np.linalg.norm(np.subtract(predictions,ytrain))))/ytrain.shape[0]

        alpha = self.params['alpha']
        err = self.params['err']

        while abs(err_w-err)>self.params['tolerance']:

            err = err_w

            if err_w>=err:
                alpha = float(alpha)/float(2)

            gra_err_w = (np.dot(Xtrain.T,(np.dot(Xtrain, iterative_weights) - ytrain))) * (1.00/float(Xtrain.shape[0]))
            iterative_weights = iterative_weights - (gra_err_w * alpha)
            self.weights = iterative_weights

            predictions = self.predict(Xtrain)
            err_w = (np.square(np.linalg.norm(np.subtract(predictions, ytrain)))) / ytrain.shape[0]


    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class LassoLinearRegression(Regressor):
    """
    Linear Regression with Stochastic Gradient Descent
    """

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'epochs': 5, 'alpha': 0.01, 'lambda': 0.0001}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        #iterative_weights = np.random.random_sample(size=Xtrain.shape[1])
        iterative_weights = np.zeros(Xtrain.shape[1])
        #iterative_weights = np.random.randint(low = 50,high=100,size=Xtrain.shape[1])
        error_list = []
        for i in range(self.params['epochs']):
            dataset = np.insert(Xtrain, Xtrain.shape[1], ytrain, axis=1)
            np.random.shuffle(dataset)
            alpha = self.params['alpha']
            for t in range (Xtrain.shape[1]):
                tempy = dataset[:, Xtrain.shape[1]]
                tempx = np.delete(dataset, Xtrain.shape[1], axis=1)
                Xtrain_t = tempx[t,:]
                ytrain_t = tempy[t]

                #if t!=0:
                    #alpha = float(self.params['alpha'])/float(math.sqrt(t))

                #err_t = alpha * np.dot(np.subtract(np.dot(Xtrain_t.T, iterative_weights),ytrain_t),Xtrain_t)
                err_t = alpha * (np.dot((np.dot(Xtrain_t.T, iterative_weights) - ytrain_t),Xtrain_t))
                iterative_weights = iterative_weights - err_t

                #Soft thresholding operation
                for index, label in enumerate(iterative_weights):

                    if label > self.params['lambda']:
                        iterative_weights[index] = label - self.params['lambda']
                    elif label <= self.params['lambda'] and label >= (-1*self.params['lambda']):
                        iterative_weights[index] = 0
                    else:
                        iterative_weights[index] = label + self.params['lambda']

            self.weights = iterative_weights

            predictions = self.predict(Xtrain)
            err_w = (np.square(np.linalg.norm(np.subtract(predictions, ytrain)))) / ytrain.shape[0]
            error_list.append(err_w)

        #plt.plot(error_list)
        #plt.ylabel('Error')
        #plt.xlabel('Epoch')
        #plt.show()

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest