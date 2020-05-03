def calculateEssentialMatrix(calibrationMatrix, fundMatrix):
    tempMatrix = np.matmul(np.matmul(calibrationMatrix.T, fundMatrix), calibrationMatrix)
    u, s, v = np.linalg.svd(tempMatrix, full_matrices=True)
    sigmaF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) # Constraining Eigenvalues to 1, 1, 0
    temp = np.matmul(u, sigmaF)
    E_matrix = np.matmul(temp, v)
    return E_matrix
