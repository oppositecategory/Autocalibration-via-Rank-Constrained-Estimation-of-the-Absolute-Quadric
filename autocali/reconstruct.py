import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint



def transform_into_matrix(upper_tri):
    # Turn the array consisting of the upper tringular part of Q into a 4x4 matrix.
    Q = np.zeros((4,4))
    inds = np.triu_indices(len(Q))
    Q[inds] = upper_tri
    Q = np.where(Q,Q,Q.T)
    print(Q)
    return Q

def objective_function(entries,cameras):
    Q = transform_into_matrix(entries)
    ws = np.array([camera.dot(Q).dot(camera.T) for camera in cameras])
    f = 0
    for w in ws:
        f += (w[0,0]-w[1,1])**2 + w[0,1]**2  + w[0,2]**2 + w[1,2]**2
    return f 

def matrix_minor(arr, i):
    # Return the i-th principal minor.
    return arr[:i,:i]

def constraint_rank_degenerate(upper_tri):
    # Return the determinant of Q.
    Q = transform_into_matrix(upper_tri)
    return np.linalg.det(Q)


def constraint_minor_det(upper_tri,i):
    # Return the determinant of the i,j minor of Q.
    Q = transform_into_matrix(upper_tri)
    minor = matrix_minor(Q,i)
    return np.linalg.det(minor)


def matrix_norm(upper_tri):
    # Return the Frobenius norm of Q
    Q = transform_into_matrix(upper_tri)
    return np.linalg.norm(Q)


def minimize_dual_quadric(cameras):
    """ Function estimates Q_infinity given the cameras matrices."""
    x0 = np.zeros((1,10))
    # Minimize function is set default such that constraint is equal to 0 and ineq means >= 0.
    cons = ({'type':'eq',   'fun':lambda q: constraint_rank_degenerate(q)},
            {'type':'eq',   'fun':lambda q: (matrix_norm(q)-1)},

            # Constraints on the principal minors of the matrix to guarantee positive-definite.
            {'type':'ineq', 'fun':lambda q: constraint_minor_det(q, 1)},
            {'type':'ineq', 'fun':lambda q: constraint_minor_det(q, 2)},
            {'type':'ineq', 'fun':lambda q: constraint_minor_det(q, 3)},
            {'type':'ineq', 'fun':lambda q: constraint_minor_det(q, 4)})
    l = 10 
    f = lambda q: objective_function(q, cameras) + l*matrix_norm(q)
    res = minimize(f,x0,method = 'Nelder-Mead',constraints=cons)
    
    if res.success:
        Q = transform_into_matrix(res.x)
        ws = np.array([camera.dot(Q).dot(camera.T) for camera in cameras])
        return Q, ws
    else:
        print ('Optimization failed.')
        print(res.message)
        return None, None


def calculate_euclidean_homography(Q):
    # Calculate the Euclidean homography by the eigendecomposition of dual absolute conic.
    sigma, v = np.linalg.eig(Q)
    return v.dot(np.sqrt(sigma))

def calculate_Ks(ws):
    Ks = []
    for w in ws:
        Ks.append(np.linalg.cholesky(w))
    return ws

