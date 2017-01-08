from __future__ import division  # floating point division
from scipy.stats import sem
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs
import plotfcns
from random import randint

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

def get_error_vector(prediction,ytest):
    return np.absolute(np.array(prediction) - np.array(ytest))

def get_residual(learner, params, selected_list, trainset):
    params['features'] = selected_list
    learner.reset(params)
    learner.learn(trainset[0], trainset[1])
    predictions = learner.predict(trainset[0])
    error_vector = get_error_vector(trainset[1], predictions)
    return error_vector


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numparams = 1
    numruns = 1


    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression': algs.FSLinearRegression({'features': range(50)}),
                'MultiSplitFSLinearRegression': algs.FSLinearRegression({'features': range(50)}),
                'RidgeRegression': algs.RidgeRegression({'features': range(50), 'lambda': 0.1}),
                'MPLinearRegression': algs.MPLinearRegression({'features': range(50)}),
                'LassoLinearRegression': algs.LassoLinearRegression({'features': range(385), 'epochs': 10, 'alpha': 0.001, 'lambda': 0.00001}),
                'StochasticGradientDescent': algs.StochasticGradientDescent({'features': range(385), 'epochs': 100, 'alpha': 0.001}),
                'BatchGradientDescent': algs.BatchGradientDescent({'features': range(385), 'alpha': 0.1, 'err': 1000000000000, 'tolerance': 0.000001})
             }


    numalgs = len(regressionalgs)

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))


    trainset, testset = dtl.load_ctscan(trainsize,testsize)
    print('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0])


    # Currently only using 1 parameter setting (the default) and 1 run
    p = 0
    r = 0
    params = {}
    threshold = 0.1
    for learnername, learner in regressionalgs.iteritems():

        # Feature Select Linear Regression
        if learnername == 'FSLinearRegression':
            print ("Running feature select regression")
            # Reset learner, and give new parameters; currently no parameters to specify
            learner.reset(params)
            # print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(testset[0])
            error = geterror(testset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)


        # Matching Pursuit Linear Regression
        if learnername == 'MPLinearRegression':
            print ("Running Matching Pursuit Linear Regression")
            selected_list = []
            learner.reset(params)
            features = learner.getparams()['features']
            selected = random.choice(features)
            selected_list.append(selected)

            error_vector = get_residual(learner, params, selected_list, trainset)
            f, corr = learner.get_feature_with_max_coeff(error_vector, trainset[0], selected_list, features)
            selected_list.append(f)
            #print corr
            while (corr > threshold):
                error_vector = get_residual(learner, params, selected_list, trainset)
                f, corr = learner.get_feature_with_max_coeff(error_vector, trainset[0],selected_list, features)
                if f is not None:
                    print corr
                    selected_list.append(f)
                else:
                    break

            params['features'] = selected_list
            learner.reset(params)
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(testset[0])
            error = geterror(testset[1], predictions)
            print len(selected_list)
            print error


        # Stochastic Gradient Descent
        if learnername == 'StochasticGradientDescent':
            print ("Running Stochastic Gradient Descent")
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(trainset[0])
            error = geterror(trainset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)



        # Batch Gradient Descent
        if learnername == 'BatchGradientDescent':
            print ("Running Batch Gradient Descent")
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(trainset[0])
            error = geterror(trainset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)


        # Lasso Regression
        if learnername == 'LassoLinearRegression':
            print ("Running Lasso Regression")
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(trainset[0])
            error = geterror(trainset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)

        # Calculating mean and standard error over multiple splits of data
        if learnername == 'MultiSplitFSLinearRegression':
            print ("Calculating mean and standard error over multiple splits of data")
            number_of_splits = 25
            error_list = []
            for i in range(number_of_splits):
                trainset, testset = dtl.load_ctscan(trainsize, testsize)
                print('Running on train={0} and test={1}...Iteration:{2}').format(trainset[0].shape[0], testset[0].shape[0],i+1)
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print 'Error: ', error
                error_list.append(error)
            mean = float(sum(error_list)) / float(len(error_list))
            std_error = sem(error_list)
            print ('Mean error: {0} and Standard error: {1}').format(mean,std_error)

        # Ridge
        if learnername == 'RidgeRegression':
            print ("Running Ridge Regression")
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(trainset[0])
            error = geterror(trainset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)



