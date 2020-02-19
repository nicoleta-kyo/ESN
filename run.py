# -*- coding: utf-8 -*-
"""
Run the ESN with NARMA input with parameters, number of runs and error measure outlined in section 3 of
     https://papers.nips.cc/paper/2318-adaptive-nonlinear-system-identification-with-echo-state-networks.pdf

"""

import numpy as np
import statistics as st
from esn import ESN
from generate_narma import gen_narma
#import warnings
#warnings.filterwarnings("error")



# ============ Network configuration and hyperparameter values ============

config = {}

# Input config

config['input_signal_type'] = 'NARMA'# input signal

# Reservoir config
 
config['n_inputs'] = 1 # nr of input dimensions
config['n_outputs'] = 1 # nr of output dimensions
config['n_reservoir'] = 100 # nr of reservoir neurons
config['spectral_radius'] = 0.8 # spectral radius of the recurrent weight matrix
config['sparsity'] = 0.95 #  proportion of recurrent weights set to zero
config['input_scaling'] = None # scaling of the input signal
config['teacher_scaling'] = None # scaling of the output signal
config['noise'] = 0.0001 # noise added to each neuron (regularization) 
config['out_activation'] = np.tanh # output activation function (applied to the readout) 
config['inverse_out_activation'] = np.arctanh # inverse of the output activation function 
config['silent'] = True # supress messages
config['augmented'] = True  # whether to use augmented states array or not

#

# ======== Set lists of different order parameters for NARMA ========
    
# List of orders
#order = [8,9,10,11]
order = [10]
#size = np.repeat(1200,4)
size_tr = [1200]
size_te = [2200]

runs = 50
runs_acc = np.zeros((len(order),runs))
# do 50 runs
for run in range(runs):
    
    print("Run " + str(run) + "...")
    
    round_acc = np.zeros((1,len(order)))
    for i in range(len(order)):
        
        # ============ Load/create input ============
        
        #create input signals
        u_tr = np.random.uniform(low = 0, high = 0.5, size = size_tr[i])
        u_te = np.random.uniform(low = 0, high = 0.5, size = size_te[i])
        
        # create NARMA input and output signals: train and test
        #input - without last element - no output for it
        d_train = u_tr[:-1]
        d_train = np.reshape(d_train, (len(d_train), 1))
        #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
        d_tr_out = gen_narma(u_tr, order[i], size_tr[i])[1:]
        
        #input - without last element - no output for it
        d_test = u_te[:-1]
        d_test = np.reshape(d_test, (len(d_test), 1))
        #output - without first element - so u and d align (otherwise d(n+1) = u(n) + ...)
        d_te_out = gen_narma(u_te, order[i], size_te[i])[1:]
        
        # if NARMA signal contains inf values, don't use it as input
        if np.isinf(d_train).any() or np.isinf(d_test).any():
            
            break
        
        # ============ Specify, train and evaluate model ============
        
        esn = ESN(
                n_inputs=config['n_inputs'],
                n_outputs=config['n_outputs'],
                n_reservoir=config['n_reservoir'],
                spectral_radius=config['spectral_radius'],
                sparsity=config['sparsity'],
                 input_scaling=config['input_scaling'],
                 teacher_scaling=config['teacher_scaling'],
                 noise=config['noise'],
                 out_activation=config['out_activation'],
                 inverse_out_activation=config['inverse_out_activation'],
                 silent = config['silent'],
                 augmented = config['augmented']
                )
        
        # train
        pred_train = esn.fit(d_train, d_tr_out, inspect=False)
        # test
        pred_test = esn.predict(d_test, continuation=False)
        #print("NARMA order: " + str(order[i]))
        
        # [nk] discard the first 200 steps when calculating the error
        # calculate mse
#        mse = np.mean((pred_test[200:] - d_te_out[200:])**2)
        mse = np.mean((pred_test[200:] - d_te_out[200:])**2)
        
        # calculate nmse
        var = st.pvariance(d_te_out[200:,0])
        nmse = mse/var
        
        #print("testing error: " + str(te_err))
        round_acc[:,i] = nmse
    
    #add results to the run
    runs_acc[:,run] = round_acc
    

# Get averages of results
#mask nan values
runs_acc[np.isnan(runs_acc)] = 0
runs_acc = np.ma.masked_equal(runs_acc, 0)
mean_err = np.mean(runs_acc, axis=1)[0]
print("Mean NMSE: " + str(mean_err))
#0.031364375
#0.309

#warnings.filterwarnings("default")














