import numpy as np

def skew(v):
    a = v[0]
    b = v[1]
    c = v[2]
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


def linear_triangulation(CameraMatrix, P0, P1, feature1, feature2): #P0 :LastColoumnOfCurrentCameraPOse
    feature1 = np.insert(np.float32(feature1), 2, 1)
    feature2 = np.insert(np.float32(feature2), 2, 1)
    # print(feature1.shape)
    homo_feature1 = np.matmul(np.linalg.inv(CameraMatrix), feature1.reshape((-1, 1)))
    homo_feature2 = np.matmul(np.linalg.inv(CameraMatrix), feature2.reshape((-1, 1)))

    skew0 = skew(homo_feature1.reshape((-1,)))
    skew1 = skew(homo_feature2.reshape((-1,)))

    P0 = np.concatenate((P0[:, :3], -np.matmul(P0[:, :3], P0[:, 3].reshape(-1, 1))), axis=1)
    P1 = np.concatenate((P1[:, :3], -np.matmul(P1[:, :3], P1[:, 3].reshape(-1, 1))), axis=1)
    # P0 = homogeneousMat(P0)
    # P1 = homogeneousMat(P1)
    pose1 = np.matmul(skew0, P0[:3, :])
    pose2 = np.matmul(skew1, P1[:3, :])

    # Solve the equation Ax=0
    A = np.concatenate((pose1, pose2), axis=0)
    u, s, vt = np.linalg.svd(A)
    X = vt[-1]
    X = X / X[3]
    return X
