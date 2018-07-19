import math
import numpy as np


def basis_transformation(xi, mu_j, variance):
    var_const = (2 * 500000)  # = 2*c, where c is a chosen const that produces best results
    norm = np.linalg.norm((xi - mu_j))  # the euclidean-norm (||.|| subscript 2)
    norm_sq = norm ** 2
    result = norm_sq / (variance * var_const)  # variance x 2*C (C=500),

    # the variance*const above
    # is just for scaling

    basis_for_xi = math.exp(-result)
    return basis_for_xi


# calculate basis functions for each sample based on its cluster's mean
def basis_process(Xdata, clusters_info, best_k):
    design_matrix = np.zeros((len(Xdata), best_k))

    # using feature 2 (aka actual temperature feature)
    var_feature = np.transpose(Xdata)[2]
    variance = np.std(var_feature) ** 2

    # using temperature feature of a cluster to describe a cluster's variance

    for j in range(0, best_k):
        mu_j = clusters_info.cluster_centers_[j]  # the "center" of a cluster
        for i in range(0, len(Xdata)):
            # run each Xdata example, xi, through gaussian basis func
            xi = Xdata[i]

            # params: ith Xdata example, mu and variance of cluster j
            phi_j_xi = basis_transformation(xi, mu_j, variance)
            design_matrix[i][j] = phi_j_xi  # filling the design matrix

    # we now have a design matrix of dimensions: (num of Xdata examples x best_k)
    Xdata = design_matrix
    return (Xdata, variance)  # returning Xdata & the variance for the basis funcs
