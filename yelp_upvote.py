from scipy import *
from numpy import *
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.io as io
from time import time

# Taken from the assignment 
def generate_data(N, d, k, sigma, seed=12231):
    random.seed(seed)
    X=randn(N,d)
    wg=zeros(1 + d)
    wg[1:k + 1] = 10 * sign(randn(k))
    eps = randn(N) * sigma
    y = X.dot(wg[1:]) + wg[0] + eps
    return (y, X, wg, eps)

# Faster Iteration to my original lasso implementation, which was excruciatingly
# slow
def solve_lasso_faster(lamb, X, y, N, d, w_init_vec):
    """ Start algorithm with w[d+1] initialized to {0,0,0 ... 0d} """
    w_tilde = copy(w_init_vec)
    w_prev = copy(w_init_vec)
    """ Start with round 0 """
    round_robin = 0
    w_0 = w_init_vec[0]
    print("Lambda provided for lasso: ", lamb)

    # Precompute Xij^2
    X_squared = X**2
    a_j_pre = X_squared.sum(axis=0)

    while round_robin < 1000 :
        for feature_j in range(0,d):
            w_prev = copy(w_tilde)
            a_j = 2 * a_j_pre[feature_j]
            c_j = 0
            w_tilde_dot_X = X.dot(w_tilde[1:])
            w_subtract_vector = X[:,feature_j] * w_tilde[feature_j+1]
            new_diff = y - w_0 - w_tilde_dot_X + w_subtract_vector
            c_j = sum(X[:,feature_j] * new_diff)
            c_j *= 2
            """ Now run the inner loop """
            if(c_j < (-1 * lamb)):
                w_tilde[feature_j+1] = (c_j + lamb)/a_j
            elif(c_j > lamb):
                w_tilde[feature_j+1] = (c_j - lamb)/a_j
            else:
                w_tilde[feature_j+1] = 0

        """ Calculate the w_0 term for the next round robin """
        w_0 = sum(y - dot(transpose(w_tilde[1:]),transpose(X)))/N
        w_tilde[0] = w_0
        new_error = max(abs(w_tilde - w_prev))
        round_robin += 1
        if new_error < .001:
            return w_tilde, w_prev, new_error,round_robin
    print("Not done even after 1000 iterations - Exiting !")
    return w_tilde,new_error,round_robin

# Implementation using C_Next
def solve_lasso_super_fast(lamb, X, y, N, d,init_w_vec):
    """ Start algorithm with w[d+1] initialized to {0,0,0 ... 0d} """
    w_tilde = copy(init_w_vec)
    w_prev = zeros(d+1)

    """ Start with round 0 """
    round_robin = 0

    """ Choose w_0 """
    w_0 = init_w_vec[0]

    print("Lambda provided for lasso: ", lamb)

    # Precompute Xij^2
    X_squared = X**2
    a_j_pre = X_squared.sum(axis=0) * 2
    c_j = 2 * ( dot(transpose(X),(y - (dot(X,w_tilde[1:]) + w_0)))  + w_tilde[1:] * a_j_pre)
    c_next = copy(c_j)

    while round_robin < 1000 :
        # Make a copy of the old vector ..
        w_prev = copy(w_tilde)
        ans = sum(c_next)

        """ Now run the inner loop """
        for idx in range(0,d):
            if(c_next[idx] < (-1 * lamb)):
                w_tilde[idx] = (c_next[idx] + lamb)/a_j_pre[idx]
            elif(c_next[idx] > lamb):
                w_tilde[idx] = (c_next[idx] - lamb)/a_j_pre[idx]
            else:
                w_tilde[idx] = 0

        # Calculate the new w_0
        w_0 = sum(y - dot(transpose(w_tilde[1:]),transpose(X)))/N
        # Update the new w_0
        w_tilde[0] = w_0
        # Create the diff vector which we'll use to compute c_next
        diff_w = w_tilde - w_prev
        diff_w_0 = w_tilde[0] - w_prev[0]
        # Compute whether we've reached a good enough precision to warrant any other iteration.
        new_error = max(abs(diff_w))
        round_robin += 1
        if new_error < .001:
            return w_tilde, w_prev, new_error,round_robin
        # Calculate c_next
        # c' = c - 2*X_t(X(delta_w) + delta_wo) + delta_w*a
        c_next = c_next - (2 * dot(transpose(X),(dot(X,diff_w[1:]) + diff_w_0)) + w_tilde[1:]*a_j_pre)
    print("Not done even after 1000 iterations - Exiting !")
    return w_tilde,new_error,round_robin

# My most performant implementation 
# Worked really well on the yelp upvote dataset
def solve_lasso_insane(lamb, X, y, N, d,init_w_vec):
    """ Start algorithm with w[d+1] initialized to {0,0,0 ... 0d} """
    w_tilde = copy(init_w_vec)
    w_prev = zeros(d+1)

    """ Start with round 0 """
    round_robin = 0

    """ Choose w_0 """
    w_0 = init_w_vec[0]

    print("Lambda provided for lasso: ", lamb)

    # Precompute Xij^2
    X_squared = X**2
    a_j_pre = X_squared.sum(axis=0) * 2

    Xw = X.dot(transpose(w_tilde[1:]))
    while round_robin < 3000 :
        # Make a copy of the old vector ..
        w_prev = copy(w_tilde)

        for feature_j in range(0,d):

            c_j = 2 * X[:,feature_j].dot(y - Xw + X[:,feature_j] * w_tilde[feature_j+1] - w_0)
            w_tilde_j_old = w_tilde[feature_j+1]
            if(c_j < (-1 * lamb)):
                w_tilde[feature_j+1] = (c_j + lamb)/a_j_pre[feature_j]
            elif(c_j > lamb):
                w_tilde[feature_j+1] = (c_j - lamb)/a_j_pre[feature_j]
            else:
                w_tilde[feature_j+1] = 0
            delta_w = w_tilde[feature_j+1] - w_tilde_j_old
            Xw += delta_w * X[:,feature_j]


        #print w_tilde

        # Calculate the new w_0
        w_0 = sum(y - dot(transpose(w_tilde[1:]),transpose(X)))/N
        # Update the new w_0
        w_tilde[0] = w_0
        # Create the diff vector which we'll use to compute c_next
        diff_w = w_tilde - w_prev
        # Compute whether we've reached a good enough precision to warrant any other iteration.
        new_error = max(abs(diff_w))
        #print "Expected Error :",new_error
        round_robin += 1
        if new_error < .5:
            return w_tilde, w_prev, new_error,round_robin
    print("Not done even after 3000 iterations - Exiting !")
    return w_tilde,new_error,round_robin

# Returns Precision/Recall/NonZeros
def getPrecision(truth,w_star,d,k):
    """ Total Number of Nonzero elements """
    """ Not counting w0 """
    total_nonzeros = len((w_star[1:]).nonzero()[0]) * 1.0
    total_correct_nonzeros = len((truth[1:] * w_star[1:]).nonzero()[0]) * 1.0
    if(total_nonzeros > 0):
        precision = total_correct_nonzeros/total_nonzeros * 1.0
    else:
        precision = -1
    recall = total_correct_nonzeros/k
    print("Total NonZeros - ", total_nonzeros)
    print("Total Correct NonZeros - ", total_correct_nonzeros)
    print("Precision: ", precision)
    print("Recall: ", recall)
    return precision,recall,total_nonzeros

# Returns Overall Error
def getOverallError(y,w_star,X,lmax):
    """ Total error figure """
    rss_error_vec = y - ( w_star[0] + dot(transpose(w_star[1:]),transpose(X)))
    total_rss_error = sum((rss_error_vec**2))
    overall_error = total_rss_error + lmax*(LA.norm(w_star[1:],1))
    return overall_error,total_rss_error

# Returns RMSE
def getRMSE(y,w_star,X,N):
    """ Total error figure """
    rss_error_vec = y - ( w_star[0] + dot(transpose(w_star[1:]),transpose(X)))
    total_rss_error = sum((rss_error_vec**2))
    overall_error = sqrt(total_rss_error/N)
    return overall_error

def getCorrectionVector(y,X,w_star):
    """ Get the vector to verify if your w_star is correct check """
    test_vec = dot(X,w_star[1:]) + w_star[0] - y
    correct_vec = dot(2 * transpose(X), test_vec)
    #print correct_vec

# Doing a single Run as opposed to finding old lamda's 
def oneRun(faster,lamb,X,y,wg,N,d,k,w_init):
    if faster == 0:
        w_star,w_prev,error,iters = solve_lasso_faster(lamb, X, y, N, d, w_init)
    else:
        w_star,w_prev,error,iters = solve_lasso_insane(lamb, X, y, N, d, w_init)

    print(wg)
    print(w_star)
    print("quit doing round robing when max difference of ", error)
    print("iters to converge: ", iters)

    """ Get total check """
    correction_vector = getCorrectionVector(y,X,w_star)
    print correction_vector

    overall_error = getOverallError(y,w_star,X,lamb)
    precision,recall,nonzero = getPrecision(wg,w_star,d,(k*1.0))
    rmse = getRMSE(y,w_star,X,N)

    print("Lambda,Error,Precision,Recall,RMSE",lamb,overall_error,precision,recall,rmse)
    return w_star

# Taken as is from Course Notes 
def gen_yelp_data():
    # Load text file of integers
    y = loadtxt("hw1-data/upvote_labels.txt",dtype=int)
    # Load text file of strings
    featurenames = open("hw1-data/upvote_features.txt").read().splitlines()
    # Load a csv of floats
    A = genfromtxt("hw1-data/upvote_data.csv",delimiter=",")
    B = io.mmio.mmread("hw1-data/star_data.mtx").tocsc()
    return y,featurenames,A,B

# Calculate LMAX
def getLMax(x,w,y):
    y_bar =  dot(x,w[1:]) + w[0]
    lmax = 2 * LA.norm(dot(transpose(x),(y - y_bar)),inf)
    return lmax

# Finds the largest weights in the final resulting vector
def findLargestWeights(w_star,names):
    nonzero_tuples = nonzero(w_star[1:])
    feature_list = []
    for idx in nonzero_tuples[0]:
        name = names[idx]
        weight = abs(w_star[idx+1])
        entry = weight,name
        feature_list.append(entry)
        print names[idx],w_star[idx-1]
    feature_list.sort()
    feature_list.reverse()
    print feature_list

def main(argv=None):

    N = 1000
    d = 1000
    k = d
    y,names,X,X2 = gen_yelp_data()
    wg = zeros(d+1)
    w_init = zeros(d+1)
    Z = X[4000:5000]
    Y = y[4000:5000]

    """
    # Check with Small DataSet
    N = 50
    d = 75
    k = 5
    sigma = 1
    seed = 12231
    y,X,wg,eps = generate_data(N,d,k,sigma,seed)
    """

    """ Calculate lmax from the synthetic data """
    """ The formula is the sup-norm(max absolute value of 2 * (X_transpose * (y-y_bar)) """
    """ y bar is just Xw + wO """
    """ However, y_bar is initially 0, since we start with vector w init to 0  """

    lmax = getLMax(X,w_init,y)
    print("Lambda Max is ", lmax)

    """ Running on Validation data """
    data = loadtxt('lambda1.0')
    w_init = data[0:1001]
    before_debias = zeros(5)
    after_debias = zeros(5)
    indices = array([1.0,27.0,54,80,106])
    idx=0

    # Print out the final weights, associated with their names
    # findLargestWeights(w_init,names)

    u,i = getOverallError(Y,w_init,Z,1.0)
    o = getRMSE(Y,w_init,Z,N)
    print u,i,o
    #return

    t0 = time()
    w_s = oneRun(1,0,Z,Y,wg,N,d,k,w_init)
    n_o = getRMSE(Y,w_s,Z,N)
    print "before_debiasing, after_debiasing:",o,n_o
    before_debias[idx] = o
    after_debias[idx] = n_o
    idx += 1

    data = loadtxt('lambda27.2631578947')
    w_init = data[0:1001]
    o = getRMSE(Y,w_init,Z,N)
    w_s = oneRun(1,0,Z,Y,wg,N,d,k,w_init)
    n_o = getRMSE(Y,w_s,Z,N)
    print "before_debiasing, after_debiasing:",o,n_o
    before_debias[idx] = o
    after_debias[idx] = n_o
    idx += 1

    data = loadtxt('lambda53.5263157895')
    w_init = data[0:1001]
    o = getRMSE(Y,w_init,Z,N)
    w_s = oneRun(1,0,Z,Y,wg,N,d,k,w_init)
    n_o = getRMSE(Y,w_s,Z,N)
    print "before_debiasing, after_debiasing:",o,n_o
    before_debias[idx] = o
    after_debias[idx] = n_o
    idx += 1

    data = loadtxt('lambda79.7894736842')
    w_init = data[0:1001]
    o = getRMSE(Y,w_init,Z,N)
    w_s = oneRun(1,0,Z,Y,wg,N,d,k,w_init)
    n_o = getRMSE(Y,w_s,Z,N)
    print "before_debiasing, after_debiasing:",o,n_o
    before_debias[idx] = o
    after_debias[idx] = n_o
    idx += 1

    data = loadtxt('lambda106.052631579')
    w_init = data[0:1001]
    o = getRMSE(Y,w_init,Z,N)
    w_s = oneRun(1,0,Z,Y,wg,N,d,k,w_init)
    n_o = getRMSE(Y,w_s,Z,N)
    print "before_debiasing, after_debiasing:",o,n_o
    before_debias[idx] = o
    after_debias[idx] = n_o

    plt.plot(indices,before_debias,'ro')
    plt.ylabel("RMSE Before Debias on Validation")
    plt.xlabel("lambda")
    plt.show()

    plt.plot(indices,after_debias,'ro')
    plt.ylabel("RMSE After Debias on Validation")
    plt.xlabel("lambda")
    plt.show()
    
    # Print Debiasing Experiment Results
    print after_debias
    print before_debias

    #for plotting
    plot_space = 31
    rmse_vals = zeros(plot_space)
    precision_vals = zeros(plot_space)
    recall_vals = zeros(plot_space)
    nonzero_vals = zeros(plot_space)
    x_axis = zeros(plot_space)
    idx=0
    w_init_vec = zeros(d+1)

    # Create lists for checking a variety of lambda's
    s1 = logspace(log2(lmax),550,num=10,base=2.0)
    s2 = linspace(1,500,20)
    s3 = concatenate((s1,s2),axis=0)

    
    for lamb in sort(s3):
        t3 = time()
        w_star,w_prev,error,iters = solve_lasso_insane(round(lamb), Z, Y, N, d, w_init_vec)
        t4 = time()
        filename = "lambda{0}".format(lamb)
        savetxt(filename,w_star)
        print(wg)
        print(w_star)

        # Get Metadata Associated
        correction_vector = getCorrectionVector(y,X,w_star)
        print("quit doing round robing when max difference of ", error)
        print("iters to converge: ", iters)
        overall_error,rss_error = getOverallError(Y,w_star,Z,lamb)
        rmse = getRMSE(y,w_star,X,N)

        precision,recall,nonzeros = getPrecision(wg,w_star,d,(k * 1.0))
        rmse_vals[idx] = rmse
        precision_vals[idx] = precision
        recall_vals[idx] = recall
        x_axis[idx] = lamb
        nonzero_vals[idx] = nonzeros
        idx += 1

        print("Lambda,Error,RSS Error,Precision,Recall,RMSE",lamb,overall_error,rss_error,precision,recall,rmse)
        print("fn took", t4-t3)
        print("----------")
        w_init_vec = copy(w_star)

    # Plotting verification
    print rmse_vals
    print precision_vals
    print recall_vals
    print x_axis
    print nonzero_vals
    plt.plot(x_axis,rmse_vals,'ro')
    plt.ylabel("RMSE on training")
    plt.xlabel("lambda")
    plt.xlim(0,500)
    plt.show()
    plt.plot(x_axis,nonzero_vals,'ro')
    plt.ylabel("NonZero Values")
    plt.xlabel("lambda")
    plt.xlim(0,500)
    plt.show()
    plt.plot(x_axis,precision_vals,'ro')
    plt.ylim(0,1)
    plt.xlim(0,lmax)
    plt.ylabel("precision")
    plt.xlabel("lambda")
    plt.show()
    plt.plot(x_axis,recall_vals,'ro')
    plt.ylabel("recall")
    plt.xlabel("lambda")
    plt.show()

if __name__ == "__main__":
    main()

