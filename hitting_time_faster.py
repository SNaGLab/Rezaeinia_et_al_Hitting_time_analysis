from joblib import Parallel, delayed
import numpy as np

def hitting_matrix_p2(correlation_matrix):
    """
    Written by Drew E. Winters, PhD. 
    Faster version of hitting time based on the function in the work of
        Rezaeinia, P., Fairley, K., Pal, P., Meyer, F. G., & Carter, R. M. (2020). 
        Identifying brain network topology changes in task processes and psychiatric disorders. 
        Network Neuroscience, 4(1), 257-273.
    """
    correlation_matrix = np.array(abs(correlation_matrix))  # Ensure absolute values
    np.fill_diagonal(correlation_matrix, 0)  # Set diagonal to 0

    L = correlation_matrix.shape[0]
    A_matrix = correlation_matrix.copy()

    # Degree matrix
    row_sums = A_matrix.sum(axis=1) # instead of d_matrix loop we sum columns without the loop
    d_max = row_sums.max()

    # Ensure graph connectivity
    for j in range(L):
      if np.max(A_matrix[j,:]) < .05:
          A_matrix[j,j] = d_max - row_sums[j]

    row_sums = A_matrix.sum(axis=1)  # Recalculate after adjustment
    D_inv = np.diag(1.0 / row_sums)
    D_sqrt = np.diag(np.sqrt(row_sums))
    D_sqrt_inv = np.diag(1.0 / np.sqrt(row_sums))

    # Transition probability matrix and Graph Laplacian
    p_matrix = D_inv @ A_matrix
    eye_P = np.eye(L) - p_matrix
    G_Lap_n = D_sqrt @ eye_P @ D_sqrt_inv

    # Eigen decomposition
    eig_val, eig_vec = np.linalg.eigh(G_Lap_n)

    # Precompute reusable quantities
    eig_val_nonzero = eig_val[eig_val > eig_val.min()]
    eig_vec_squared = eig_vec ** 2
    d_total = row_sums.sum()

    def compute_H_row(i):
        H_row = np.zeros(L)
        deg_i = row_sums[i]
        for j in range(L):
            deg_j = row_sums[j]
            t_ij = (
                eig_vec_squared[i, eig_val > eig_val.min()] / deg_i
                - eig_vec[i, eig_val > eig_val.min()]
                * eig_vec[j, eig_val > eig_val.min()]
                / np.sqrt(deg_i * deg_j)
            )
            H_row[j] = np.sum(d_total * t_ij / eig_val_nonzero)
        return H_row

    # Parallelize computation of rows
    with Parallel(n_jobs=-1, backend="loky") as parallel:
        H = np.array(parallel(delayed(compute_H_row)(i) for i in range(L)))
    return H

