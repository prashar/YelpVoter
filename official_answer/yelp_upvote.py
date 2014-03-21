#!/usr/bin/python2
from scipy import *
import numpy as np
import scipy.sparse as sp
from lasso import *
from yelp import *

UX = genfromtxt("data/upvote_data.csv", delimiter=",")
Ut = loadtxt("data/upvote_labels.txt", dtype=np.float)
Ul = open("data/upvote_features.txt").read().splitlines()

Atr = UX[:4000, :].copy()
btr = Ut[:4000].copy()

Av = UX[4000:5000, :].copy()
bv = Ut[4000:5000].copy()

Ats = UX[5000:, :].copy()
bts = Ut[5000:].copy()

stats = yelp_solve((Atr, btr), (Av, bv), (Ats, bts), Ul)


(ret, res) = stats

#######################################################
plt.clf()
plt.plot(log10(ret[:, 0]), ret[:, 1], 'o-', label="Non-zeros")
plt.xlabel("$\\log_{10}(\\lambda)\\rightarrow$")
plt.ylabel("$N_{z}$")
plt.legend()

#######################################################
plt.figure()
plt.subplot(121)
plt.plot(log10(ret[:, 0]), sqrt(ret[:, 2]/Av.shape[0]), 'o-', label="RMSE, Validation")
plt.plot(log10(ret[:, 0]), sqrt(ret[:, 4]/Atr.shape[0]), 'o-', label="RMSE, Training")
plt.xlabel("$\\log_{10}(\\lambda)\\rightarrow$")
plt.ylabel("$\\epsilon_{RMSE}$")
plt.legend()
axs = plt.axis()

plt.subplot(122)
plt.plot(log10(ret[:, 0]), sqrt(ret[:, 3]/Av.shape[0]), 'o-', label="Debiased RMSE, Validation")
plt.plot(log10(ret[:, 0]), sqrt(ret[:, 5]/Atr.shape[0]), 'o-', label="Debiased RMSE, Training")
plt.axis(axs)
plt.xlabel("$\\log_{10}(\\lambda)\\rightarrow$")
plt.ylabel("$\\epsilon_{RMSE}$")
plt.legend()

#######################################################
plt.figure()

wva = array(map(lambda x: x[1], res))

plt.subplot(121)
plt.plot(log10(ret[:, 0]), wva, label="Weights")
plt.xlabel("$\\log_{10}(\\lambda)\\rightarrow$")

midx = argmin(ret[:, 2])
wv_asort = argsort(-abs(wva[midx, 1:]))

rmse = sqrt(((Ats.dot(wva[midx, 1:]) + wva[midx, 0] - bts)**2).mean())
top_l = [Ul[i] for i in list(wv_asort[:10])]

wdb = array(map(lambda x: x[2], res))
plt.subplot(122)
plt.plot(log10(ret[:, 0]), wdb, label="Weights")
plt.xlabel("$\\log_{10}(\\lambda)\\rightarrow$")

midx_db = argmin(ret[:, 3])
wd_asort = argsort(-abs(wdb[midx_db, 1:]))

rmse_db = sqrt(((Ats.dot(wdb[midx_db, 1:]) + wdb[midx_db, 0] - bts)**2).mean())
top_l_db = [Ul[i] for i in list(wd_asort[:10])]

for i in range(10):
    print i + 1, "&", top_l[i], ", ", "%.2f" % wva[midx, 1 + wv_asort[i]], "&", top_l_db[i], ", ", "%.2f" % wdb[midx_db, 1 + wd_asort[i]], "\\\\"
