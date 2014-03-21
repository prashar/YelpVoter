from scipy import *
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.io as io
import scipy.sparse as sp
import lasso as ls
import data_generator as dg

def prob_5_2_1(N, d, k, sigma, rat=9/10., seed=12231):
    # N = 50, d = 75, k = 5, sigma = 1
    
    (y, X, wg, eps) = dg.generate_data(N, d, k, sigma, seed)
    ##########
    w = randn(1 + d)
    
    prob = ls.initialize_computation(X, w, y)
    ret = []
    coeffs = []
    for i in range(100):
        ls.round_robin(prob, 1000)
        ret.append([prob[3], sum(abs(prob[0][1:k + 1]) > 0), sum(abs(prob[0][1:]) > 0)])
        coeffs.append(prob[0].copy())
        prob[3] *= rat
    ret = array(ret)
    coeffs = array(coeffs)
    #Precision
    ret[:, 2] = ret[:, 1]/ret[:, 2]
    #Recall
    ret[:, 1] = ret[:, 1]/float(k)

    return (prob, array(ret), coeffs)

def prob_5_2_2(N, d, k, sigma, lam = 300, seed=12231):
    # N = 50, d = 75, k = 5, sigma = 10
    #lambda^* = 300
    (y, X, wg, eps) = dg.generate_data(N, d, k, sigma, seed)
    ##########
    w = randn(1 + d)    
    prob = ls.initialize_computation(X, w, y)
    prob[3] = lam
    ls.round_robin(prob, 1000)
    return (prob, sum(abs(prob[0][1:k + 1]) > 0), sum(abs(prob[0][1:]) > 0))
