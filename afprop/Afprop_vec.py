import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
import cython
from numba import jit


def calc_similarity_matrix(mydata, num_cluster_pref=1):
    neg_euc_dist = -cdist(mydata, mydata, "euclidean") ** 2
    pref = np.min(neg_euc_dist) if num_cluster_pref == 1 else np.median(neg_euc_dist)
    np.fill_diagonal(neg_euc_dist, pref)
    return neg_euc_dist


def init_r_array(s_matrix):
    """compute responsibility matrix for iteration 0"""
    r_array_0 = s_matrix - (s_matrix).max(axis=1)[:, None]
    return r_array_0


def r_array_update(niter, a_array, s_matrix, damp_c, r_array):
    """compute responsibility matrix for iteration #niter"""
    s_a_array_sum = a_array[niter] + s_matrix
    n = s_a_array_sum.shape[1]
    row_max = np.zeros((n, n))
    rng = np.arange(n)
    for i in rng:
        row_max[:, i] = np.amax(s_a_array_sum[:, rng != i], axis=1)
    update_term = s_matrix - row_max
    r_array[niter] = damp_c * update_term + (1 - damp_c) * r_array[niter - 1]
    return r_array[niter]


def a_array_update(num_data_pts, niter, r_array, damp_c, a_array):
    """compute availiability for iteration #niter"""
    for i in range(num_data_pts):
        for k in range(num_data_pts):
            if i != k:
                temp_vec = np.ones(num_data_pts)
                temp_vec[k] = 0
                temp_vec[i] = 0
                temp_vec[r_array[niter - 1, :, k] < 0] = 0
                a_ik_sum = np.dot(r_array[niter - 1, :, k], temp_vec)
                for_min = np.concatenate(
                    (r_array[niter - 1, k, k] + a_ik_sum, np.zeros(1)), axis=None
                )
                update_term = np.min(for_min)
                a_array[niter, i, k] = (
                    damp_c * update_term + (1 - damp_c) * a_array[niter - 1, i, k]
                )
            else:
                temp_vec = np.ones(num_data_pts)
                temp_vec[k] = 0
                temp_vec[r_array[niter - 1, :, k] < 0] = 0
                a_ik_sum = np.dot(r_array[niter - 1, :, k], temp_vec)
                update_term = a_ik_sum
                a_array[niter, i, k] = (
                    damp_c * update_term + (1 - damp_c) * a_array[niter - 1, i, k]
                )
    return a_array[niter]


def make_cluster_plot(mydata, r_s_sum_array, is_center):
    """scatter plot of clusters with centers labeled"""
    if mydata.shape[1] == 2:
        plt.scatter(
            mydata[:, 0], mydata[:, 1], c=np.argmax(r_s_sum_array, axis=1), s=200,
        )
        plt.scatter(
            mydata[:, 0][is_center],
            mydata[:, 1][is_center],
            marker="+",
            s=350,
            c="black",
        )
        plt.show()


def afprop_vec(
    mydata, num_cluster_pref=1, iterations=100, damp_c=0.5, num_stable_iters=10
):
    """
    Perform clustering by affinity propagation
    
    Parameters
    ----------
    
    mydata : array-like, shape (num_data_pts, 2)
        Data set for clustering.
        
    num_cluster_pref : int (1 or 2), optional, default: 1
        Indication of whether the input similarities are set to the minimum similarity measure 
        (IE, negative squared Euclidean distance) between the data points (num_cluster_pref = 1), 
        or the median similarity measure (num_cluster_pref = 2).
        
    iterations : int, optional, default: 100
        Maximum number of iterations to run affinity propagation.
    
    damp_c : float, optional, default: 0.5
        Damping constant. The weight given to the updating iteration; between 0 and 1.
        
    num_stable_iters : int, optional, default: 10
        The algorithm stops if the clusterings at each iteration have remained unchanged for 
        this many iterations in a row.
        
       
    Returns
    -------
    
    clusters : array, shape (num_data_pts, 1)
        For the index of each data point, gives that data point's exemplar.
    
    exemplars : array, shape(num_exemplars, 1)
        List the indices of the data points that are exemplars.
    
    cluster_plot : figure
        Scatter plot of the data, with points in the same cluster hacing the same color, and
        exemplars marked with plus-signs.
        
    num_clusters : int
        Number of clusters in final clustering.
        
    final_iter : int
        Number of iterations that were run before the algorithm stopped.
        
        
    Notes
    -----
    
    Input preferences should be set to the minimum similarity measure when a small number of clusters
    is preferred, and to the median similarity measure when a moderate number of clusters is 
    preferred.
    
    If stability is not acheived, the aglorithm does not output the returned objects but 
    rather prints an error message: "Stability not acheived. Consider reducing num_stable_iters.""""
    # convert pd to np array
    if isinstance(mydata, pd.DataFrame):
        mydata = mydata.values

    # data input error messages
    if mydata.shape[1] != 2:
        raise ValueError("S must have two columns.")
    if num_cluster_pref != 1 and num_cluster_pref != 2:
        raise ValueError(
            "Enter valid indication (1 or 2) of cluster number preference."
        )
    if iterations < 1 or type(iterations) != int:
        raise ValueError("Enter a valid number of iterations.")
    if damp_c <= 0 or damp_c > 1:
        raise ValueError("Enter a valid damping constant.")
    if (
        num_stable_iters < 1
        or num_stable_iters > iterations
        or type(num_stable_iters) != int
    ):
        raise ValueError("Enter a valid number of iterations to check for stability.")

    # count number of data points, IE number rows in mydata
    num_data_pts = mydata.shape[0]

    s_matrix = calc_similarity_matrix(mydata, num_cluster_pref=1)

    # initialize a_array: a(i,k) = 0 at 0th iteration
    a_array = np.zeros(num_data_pts * num_data_pts * (iterations)).reshape(
        (iterations, num_data_pts, num_data_pts)
    )

    # initialize r_array
    r_array = np.zeros((iterations, num_data_pts, num_data_pts))

    # fill in r_array values for 0th iteration
    r_array[0] = init_r_array(s_matrix)

    ### iterative loop for iterations 1+

    # define tracker variables for checking for stability
    clusters_prev = np.zeros(num_data_pts)
    iter_stability = np.zeros(iterations)

    for niter in range(1, iterations):

        # update a and r arrays at each iteration
        a_array[niter] = a_array_update(num_data_pts, niter, r_array, damp_c, a_array)
        # r_array[niter] = r_array_update(num_data_pts, niter, a_array, s_matrix, damp_c, r_array)
        r_array[niter] = r_array_update(niter, a_array, s_matrix, damp_c, r_array)
        r_s_sum_array = r_array[niter] + a_array[niter]

        # results of each iteration's clustering attempt
        clusters = np.argmax(
            r_s_sum_array, axis=1
        )  # the list points grouped by their assigned center
        centers = np.where(
            np.argmax(r_s_sum_array, axis=1) == np.array(range(num_data_pts))
        )  # the points that are centers
        is_center = np.argmax(r_s_sum_array, axis=1) == np.array(
            range(num_data_pts)
        )  # true if pt is a center, false otherwise

        # record whether this iteration's clustering is the same as in previous iteration
        if np.array_equal(clusters, clusters_prev):
            iter_stability[niter] = 1

        clusters_prev = clusters

        # if you have seen enough identical clusterings in a row,
        # create a scatterplot illustrating the results
        # and break the iteration loop and return the final clustering results
        if niter > num_stable_iters and np.all(
            iter_stability[niter - num_stable_iters : niter] == 1
        ):

            centers = np.where(
                np.argmax(r_s_sum_array, axis=1) == np.array(range(num_data_pts))
            )  # the points that are centers
            is_center = np.argmax(r_s_sum_array, axis=1) == np.array(
                range(num_data_pts)
            )  # true if pt is a center, false otherwise

            exemplars = centers[0]
            num_clusters = len(np.unique(clusters))
            final_iter = niter + 1

            # if data is 2D, print scatter plot
            if mydata.shape[1] == 2:
                make_cluster_plot(mydata, r_s_sum_array, is_center)
                return clusters, exemplars, num_clusters, final_iter
            else:
                return clusters, exemplars, num_clusters, final_iter
            break

        # if you have not seen enough identical clusterings in a row by the final iteration,
        # just print a message
        elif niter == iterations - 1:
            print("Stability not acheived. Consider reducing num_stable_iters.")
