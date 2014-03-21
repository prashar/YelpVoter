#!/usr/bin/python2
from scipy import *
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from lasso import *

def yelp_solve(train, val, tes, Ul):
    (Atr, btr) = train
    (Av, bv) = val
    (Ats, bts) = tes

    prob = initialize_computation(Atr, zeros(Atr.shape[1] + 1), btr)
    #lmax/2
    prob[3] /= 2

    ret = []
    res = []
    for i in range(100):
        round_robin(prob, 1000, 1e-3, 1e-3)
        nzeros = sum(abs(prob[0][1:]) > 0)
        trerr = ((Atr.dot(prob[0][1:]) + prob[0][0] - btr)**2).sum()
        verror = ((Av.dot(prob[0][1:]) + prob[0][0] - bv)**2).sum()
        #
        tmp_w = prob[0].copy()
        tmp_c = prob[5].copy()
        #4
        try:
            debias_nz(prob)
        except la.LinAlgError:
            update_nz(prob, 1000, 1e-3, 1e-3)

        trerr_db = ((Atr.dot(prob[0][1:]) + prob[0][0] - btr)**2).sum()
        verror_db = ((Av.dot(prob[0][1:]) + prob[0][0] - bv)**2).sum()
        #
        lam = prob[3]
        ret.append([lam, nzeros, verror, verror_db, trerr, trerr_db])
        res.append([lam, tmp_w.copy(), prob[0].copy()])
        #Restore the class.
        prob[0] = tmp_w
        prob[5] = tmp_c
        #
        prob[3] /= 2
        if(prob[3] < 1e-1):
            break
    ret = array(ret)

    return (ret, res)
