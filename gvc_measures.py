import numpy as np

def upstreamness(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Create identity matrix
    I = np.identity(A.shape[0])
    # Creates 1's vector
    ones = np.ones(A.shape[0])
    # Calculate inverse y diagonal matrix
    y_inv = np.divide(1, y, out=np.zeros(y.shape), where=y!=0)
    Y_inv = np.diag(y_inv)
    # Calculate Au matrix
    Au = (Y_inv@A)
    # Calculate upstreamness vector
    U = np.linalg.inv(I - Au)@ones
    return U

def downstreamness(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Create identity matrix
    I = np.identity(A.shape[0])
    # Creates 1's vector
    ones = np.ones(A.shape[0])
    # Calculate inverse y diagonal matrix
    y_inv = np.divide(1, y, out=np.zeros(y.shape), where=y!=0)
    Y_inv = np.diag(y_inv)
    # Calculate Ad matrix
    Ad = (A@Y_inv).T
    # Calculate downstreamness vector 
    D = np.linalg.inv(I - Ad)@ones
    return D

def upstreamness_rank1(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Calculate inverse y diagonal matrix
    y_inv = np.divide(1, y, out=np.zeros(y.shape), where=y!=0)
    Y_inv = np.diag(y_inv)
    # Calculate Au matrix
    Au = (Y_inv@A)
    # row sum of Au
    r = Au.sum(axis=1)
    # upstreamness rank 1 estimate
    U_rank1 = 1 + r/(1 - np.mean(r))
    return U_rank1 

def downstreamness_rank1(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Calculate inverse y diagonal matrix
    y_inv = np.divide(1, y, out=np.zeros(y.shape), where=y!=0)
    Y_inv = np.diag(y_inv)
    # Calculate Ad matrix
    Ad = (A@Y_inv).T
    # row sum of Ad
    r = Ad.sum(axis=1)
    # downstreamness rank 1 estimate
    D_rank1 = 1 + r/(1 - np.mean(r))
    return D_rank1 