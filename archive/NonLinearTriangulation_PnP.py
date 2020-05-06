
import numpy as np
from scipy.optimize import least_squares


def nonLinearTriangulationError(X, P1, P2, x1, x2, K):  # X is the output of linear triangular
    Xpt = np.reshape(X, (np.shape(X)[0], 1))
    P1 = np.concatenate((P1[:, :3], -np.matmul(P1[:, :3], P1[:, 3].reshape(-1, 1))), axis=1)
    P2 = np.concatenate((P2[:, :3], -np.matmul(P2[:, :3], P2[:, 3].reshape(-1, 1))), axis=1)
    P1 = np.matmul(K, P1)
    P2 = np.matmul(K, P2)
    err1 = (x1[0] - (np.dot(P1[0], Xpt) / np.dot(P1[2], Xpt)))**2 + (x1[1] - (np.dot(P1[1], Xpt) / np.dot(P1[2], Xpt)))**2
    err2 = (x2[0] - (np.dot(P2[0], Xpt) / np.dot(P2[2], Xpt)))**2 + (x2[1] - (np.dot(P2[1], Xpt) / np.dot(P2[2], Xpt)))**2

    err = err1 + err2
    return err


def NonlinearTriangulation(K, pose1, pose2, pts0, pts1, X): 
    #print("outside function",X[0])
    Xinit = np.reshape(X, (np.shape(X)[0],))
    filteredPoints = least_squares(nonLinearTriangulationError,
                                   x0=Xinit,
                                   args=(pose1, pose2, pts0, pts1, K), ftol=0.001, xtol=0.001, gtol=0.001, max_nfev=1000)
    return filteredPoints.x / filteredPoints.x[3]


def residual(P, X, x, K):  
    P = np.reshape(P, (3, 4))
    P = np.matmul(K, P)
    sum = 0
    for pt in range(len(X)):
        Xpt = np.reshape(X[pt], (4, 1))
        err = (x[pt, 0] - (np.dot(P[0], Xpt) / np.dot(P[2], Xpt)))**2 + (x[pt, 1] - (np.dot(P[1], Xpt) / np.dot(P[2], Xpt)))**2
        sum = sum + err
    return sum


def nonlinearPnP(X, x, K, Cnew, Rnew): #x is pts2 returned by features

    t = np.array([[1, 0, 0, -Cnew[0]],
                  [0, 1, 0, -Cnew[1]],
                  [0, 0, 1, -Cnew[2]]])
    P = np.matmul(Rnew, t)
    P_init = np.reshape(P, (P.shape[0] * P.shape[1],))
    res = least_squares(residual, P_init, args=(X, x, K), ftol=0.001, xtol=0.001, gtol=0.001, max_nfev=1000)
    P_refined = res.x

    P_refined = np.reshape(P_refined, (3, 4))
    R_refined = P_refined[:, 0:3]
    U, S, Vh = np.linalg.svd(R_refined)
    print(S)
    R_refined = np.reshape(np.matmul(U, Vh), (3, 3))

    C_refined = np.reshape(np.matmul(-R_refined.T, P_refined[:, 3]), (3, 1))

    print(np.linalg.det(R_refined))

    return R_refined, C_refined
