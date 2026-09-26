import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here

    # Step 1 symmetric AT_A
    AT_A = [[A[0][0]**2 + A[1][0]**2, A[0][0] * A[0][1] + A[1][0] * A[1][1]],\
            [A[0][0]*A[0][1] + A[1][0]*A[1][1], A[0][1]**2 + A[1][1]**2 ]]

    # Step 2 Jacobi rotation
    theta = 0.0
    if AT_A[0][0] == AT_A[1][1]:
        theta = np.pi / 4
    else:
        theta = 1/2 * np.arctan((2*AT_A[0][1] / (AT_A[0][0] - AT_A[1][1])))

    # Step 3 sinuglar values
    R = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    D = R.T @ AT_A @ R
    sigma_1, sigma_2 = np.sqrt(D[0][0]), np.sqrt(D[1][1])

    # Step 4 compute U
    U = A @ R @ np.array([[1/sigma_1, 0], [0, 1/sigma_2]])
    S = np.array([sigma_1, sigma_2])
    S.sort()
    
    return U, S[::-1], R.T


    
