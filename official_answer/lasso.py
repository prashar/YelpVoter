#!/usr/bin/python2

from scipy import *
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.io as io
import scipy.sparse as sp

#See what happens when a lisper writes Python :]
def initialize_computation(X, w, y, lam=None):
    #yp = X.dot(w[1:]) + w[0]

    #Number of features
    nf = len(w)
    #Number of data points
    nd = X.shape[0]

    #Compute 2 ([1; X]^T [1; X])
    ipro = zeros([nf, nf])
    tmp = X.T.dot(X)
    if(sp.issparse(X)):
        tmp = tmp.todense()
    ipro[1:, 1:] = tmp
    ipro[0, 0] = nd

    tmp = X.sum(axis = 0)
    if(sp.issparse(X)):
        tmp = array(tmp)[0]
    #The matrix is symmetric.
    ipro[0, 1:] = ipro[1:, 0] = tmp
    ipro *= 2.0

    #a_i
    a_i = diag(ipro).copy()

    #c_i
    c_i = zeros(w.shape)
    err = y - (w[0] + X.dot(w[1:]))
    lmax = 2.0 * max(abs(X.T.dot(err)))
    if lam == None:
        lam = lmax

    c_i[0] = 2.0 * err.sum() + a_i[0] * w[0]
    c_i[1:] = 2.0 * X.T.dot(err) + a_i[1:] * w[1:]

    return [w.copy(), X.copy(), y.copy(), lam, a_i, c_i, ipro]

def update_weights(prob, i):
    #All the things one does to avoid using "self." (!)
    (w, X, y, lam, a_i, c_i, ipro) = prob
    #Compute the co-ordinate step
    if(i == 0):
        wp = c_i[0]/a_i[0]
    else:
        if(c_i[i] < -lam):
            wp = (c_i[i] + lam)/a_i[i]
        elif(c_i[i] > lam):
            wp = (c_i[i] - lam)/a_i[i]
        else:
            wp = 0
    #Update c_i
    dw = (wp - w[i])
    c_i -= ipro[:, i] * dw
    c_i[i] += a_i[i] * dw

    w[i] = wp
    return (wp, dw)

def round_robin(prob, nmax, atol = 1e-3, rtol=1e-3):
    for i in range(nmax):
        abs_err = 0.;
        rel_err = 0.;
        for j in range(len(prob[0])):
            (wp, dw) = update_weights(prob, j)
            abs_err = max(abs(dw), abs_err)
            if wp != 0:
                rel_err = max(abs(dw)/wp, rel_err)
        if (abs_err <= atol) and (rel_err <= rtol):
            break;
    print("Optimization took " + str(i + 1) + " iterations. Last |dw| = " + str(abs_err) + ".")
    return

#Debiasing via co-ordinate descent (slow!)
def update_nz(prob, nmax, atol = 1e-3, rtol=1e-3):
    (w, X, y, lam, a_i, c_i, ipro) = prob
    #Set lambda to zero for debiasing.
    lam_old = lam
    prob[3] = 0.0
    for i in range(nmax):
        abs_err = 0.;
        rel_err = 0.;
        for j in range(len(w)):
            if (abs(w[j]) > 0) or j == 0:
                (wp, dw) = update_weights(prob, j)
                abs_err = max(abs(dw), abs_err)
                if wp != 0:
                    rel_err = max(abs(dw)/wp, rel_err)
        if (abs_err <= atol) and (rel_err <= rtol):
            break;
    prob[3] = lam_old
    print("Optimization took " + str(i + 1) + " iterations. Last |dw| = " + str(abs(abs_err)) + ".")
    return

#Analytical solution
def debias_nz(prob, reg=1e-3):
    (w, X, y, lam, a_i, c_i, ipro) = prob
    nd = X.shape[0]
    #This is a fancy way of pruning zeros. You could also do this with loops without too much worry. 
    #You could also use numpy.nonzero to do this.
    stn = array([0] + filter(lambda x : x != 0, list((w[1:] != 0) * range(1, len(w)))), dtype=int)
    Hn = ipro[:, stn]
    Hn = Hn[stn, :].copy()
    #
    nz = len(stn)
    #
    Xty = zeros(nz)
    Xty[0] = y.sum()
    for i in range(1, nz):
        if sp.issparse(X):
            Xty[i] = X[:, stn[i] - 1].T.dot(y)[0]
        else:
            Xty[i] = X[:, stn[i] - 1].dot(y)
    Xty *= 2
    #
    try:
        wdb = la.solve(Hn, Xty, sym_pos=True)
    except la.LinAlgError:
        print("Oh no! Matrix is Singular. Trying again using regularization.")
        Hn[range(nz), range(nz)] += 2 * reg
        wdb = la.solve(Hn, Xty, sym_pos=True)
    
    #Update c_i
    c_i -= ipro[:, stn].dot(wdb - w[stn])
    c_i[stn] += a_i[stn] * (wdb - w[stn])

    w[stn] = wdb

    return
