def fundamentalMatrix(corners1, corners2):
    A = np.empty((8, 9))

    for i in range(0, len(corners1)): # Looping over all the 8-points (features)
        x1 = corners1[i][0] # x-coordinate from current frame 
        y1 = corners1[i][1] # y-coordinate from current frame
        x2 = corners2[i][0] # x-coordinate from next frame
        y2 = corners2[i][1] # y-coordinate from next frame
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)  # Taking SVD of the matrix
    f = v[-1].reshape(3,3) # Last column of V matrix
    
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    F = u1 @ s2 @ v1
