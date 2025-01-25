# Develop generalized minimal residual method

import numpy as np

def gmres(A, b, k):

    #Checking if dimension of A is in sync with B
    m, n = A.shape
    assert m == n, "Matrix A must be square"
    assert len(b) == m, "Dimension of b must match A"

    Q = np.zeros((m, k + 1))  # Arnoldi basis
    H = np.zeros((k + 1, k))  # Upper Hessenberg matrix

    # Initialize
    x0 = np.zeros_like(b)
    r0 = b - np.matmul(A, x0)
    beta = np.linalg.norm(r0)
    if beta == 0:
        return x0  # Exact solution already

    # Start Arnoldi process
    Q[:, 0] = r0 / beta

    for j in range(k):
        # Compute the next Krylov vector
        v = np.matmul(A, Q[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], v)
            v -= H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        if H[j + 1, j] != 0 and j + 1 < k:
            Q[:, j + 1] = v / H[j + 1, j]
        else:
            break

    # Solve least-squares problem for y
    e1 = np.zeros(k + 1)
    e1[0] = beta
    y, _, _, _ = np.linalg.lstsq(H[:j+2, :j+1], e1[:j+2], rcond=None)

    # Compute approximate solution
    x_k = x0 + np.matmul(Q[:, :j+1], y)
    return x_k


# Test the function

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

# Number of iterations equal to dimension
k = len(b)

# Solve using GMRES
x_gmres = gmres(A, b, k)

# Solve using numpy.linalg.solve for comparison
x_direct = np.linalg.solve(A, b)

print("GMRES solution:", x_gmres)
print("Linalg solution:", x_direct)
print("Difference:", np.linalg.norm(x_gmres - x_direct))
