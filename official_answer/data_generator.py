##########################################################################
from scipy import *

def generate_data(N, d, k, sigma, seed=12231):
    """
    (y, X, w_ori, eps) = generate_data(N, d, k, sigma, seed=12231)

    Generate data for linear regression.
    y = (X * w_ori[1:] + w_ori[0]) + eps

    where 
    X_{ij} ~ N(0, 1)
    w_ori[0] = 0, w_ori[1:d+1] = +/- 1
    eps ~ N(0, sigma^2)
    """
    random.seed(seed)

    X = randn(N, d)

    wg = zeros(1 + d)
    wg[1:k + 1] = 10 * sign(randn(k))
    eps = randn(N) * sigma
    y = X.dot(wg[1:]) + wg[0] + eps
    
    return (y, X, wg, eps)
##########################################################################
