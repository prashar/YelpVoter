from scipy import *
from numpy import *
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.io as io
from time import time

##########
# The main routine call for figuring out sparse coefficients
##########

def solve_lasso_sparse(lamb, X_m, y, N, d,init_w_vec):
    """ Start algorithm with w[d+1] initialized to {0,0,0 ... 0d} """
    w_tilde = copy(init_w_vec)
    w_prev = zeros(d+1)

    """ Start with round 0 """
    round_robin = 0

    """ Choose w_0 """
    w_0 = init_w_vec[0]

    print("Lambda provided for lasso: ", lamb)

    # Precompute Xij^2
    # Flatten X
    X_arr = X_m.copy()
    X_arr.data **= 2

    # This variable will be updated whenever we change w
    Xw = X_m.dot(transpose(w_tilde[1:]))
    while round_robin < 3000 :
        # Make a copy of the old vector ..
        w_prev = copy(w_tilde)

        for feature_j in range(0,d):

            # convert feature j column to an array
            arr = array(X_m[:,feature_j].todense()).flatten()
            c_j = 2 * arr.dot(y - Xw + arr * w_tilde[feature_j+1] - w_0)
            w_tilde_j_old = w_tilde[feature_j+1]
            if(c_j < (-1 * lamb)):
                w_tilde[feature_j+1] = (c_j + lamb)/(2 *sum(X_m[:,feature_j].data))
            elif(c_j > lamb):
                w_tilde[feature_j+1] = (c_j - lamb)/(2 *sum(X_m[:,feature_j].data))
            else:
                w_tilde[feature_j+1] = 0
            delta_w = w_tilde[feature_j+1] - w_tilde_j_old
            Xw += delta_w * arr

        # Calculate the new w_0
        w_0 = sum(y - X_m.dot(w_tilde[1:]))/N
        # Update the new w_0
        w_tilde[0] = w_0
        # Create the diff vector which we'll use to compute c_next
        diff_w = w_tilde - w_prev
        # Compute whether we've reached a good enough precision to warrant any other iteration.
        new_error = max(abs(diff_w))
        #print "Expected Error :",new_error
        round_robin += 1
        if new_error < .001:
            return w_tilde, w_prev, new_error,round_robin
    print("Not done even after 3000 iterations - Exiting !")
    return w_tilde,new_error,round_robin


#####
#HELPER FUNCTIONS
#####

def getOverallError(y,w_star,X,lmax):
    """ Total error figure """
    rss_error_vec = y - ( w_star[0] + X.dot(transpose(w_star[1:])))
    total_rss_error = sum((rss_error_vec**2))
    overall_error = total_rss_error + lmax*(LA.norm(w_star[1:],1))
    return overall_error,total_rss_error

def getRMSE(y,w_star,X,N):
    """ Total error figure """
    #rss_error_vec = y - ( w_star[0] + dot(transpose(w_star[1:]),transpose(X)))
    rss_error_vec = y - ( w_star[0] + X.dot(transpose(w_star[1:])))
    total_rss_error = sum((rss_error_vec**2))
    overall_error = sqrt(total_rss_error/N)
    return overall_error


def getCorrectionVector(y,X,w_star):
    """ Get the vector to verify if your w_star is correct check """
    test_vec = dot(X,w_star[1:]) + w_star[0] - y
    correct_vec = dot(2 * transpose(X), test_vec)
    #print correct_vec

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

def gen_yelp_data_feature_review():
    # Load text file of integers
    y = loadtxt("hw1-data/star_labels.txt",dtype=int)
    # Load text file of strings
    featurenames = open("hw1-data/star_features.txt").read().splitlines()
    # Load a csv of floats
    X = io.mmio.mmread("hw1-data/star_data.mtx").tocsc()
    #X_arr = A.todense()
    #X = array(X_arr)
    return y,featurenames,X

def getLMax(x,w,y):
    y_bar =  dot(x,w[1:]) + w[0]
    lmax = 2 * LA.norm(dot(transpose(x),(y - y_bar)),inf)
    return lmax

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

    N = 30000
    d = 2500
    k = d
    y,names,X = gen_yelp_data_feature_review()

    w_init = zeros(d+1)
    wg = zeros(d+1)
    Z = X[0:30000]
    Y = y[0:30000]

    #for plotting
    plot_space = 5
    rmse_vals = zeros(plot_space)
    precision_vals = zeros(plot_space)
    recall_vals = zeros(plot_space)
    nonzero_vals = zeros(plot_space)
    x_axis = zeros(plot_space)
    idx=0
    w_init_vec = zeros(d+1)

    s2 = linspace(80,10,5)

    for lamb in sort(s2):
        t3 = time()
        w_star,w_prev,error,iters = solve_lasso_sparse(round(lamb), Z, Y, N, d, w_init_vec)
        t4 = time()
        filename = "lambda{0}".format(lamb)
        savetxt(filename,w_star)
        print(w_star)

        #Get total check
        #correction_vector = getCorrectionVector(y,X,w_star)
        #print correction_vector
        #print("quit doing round robing when max difference of ", error)
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

    t4 = time()
    print("fn took", t4-t3)

    # Plotting verification
    print rmse_vals
    print precision_vals
    print recall_vals
    print x_axis
    print nonzero_vals
    plt.plot(x_axis,rmse_vals,'ro')
    plt.ylabel("RMSE on training")
    plt.xlabel("lambda")
    plt.xlim(0,90)
    plt.show()
    plt.plot(x_axis,nonzero_vals,'ro')
    plt.ylabel("NonZero Values")
    plt.xlabel("lambda")
    plt.xlim(0,90)
    plt.show()
    plt.plot(x_axis,precision_vals,'ro')
    plt.ylim(0,1)
    plt.xlim(0,90)
    plt.ylabel("precision")
    plt.xlabel("lambda")
    plt.show()
    plt.plot(x_axis,recall_vals,'ro')
    plt.ylabel("recall")
    plt.xlabel("lambda")
    plt.show()

if __name__ == "__main__":
    main()

