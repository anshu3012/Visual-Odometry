def CameraPoseE(essentialMatrix):
    u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 1st Solution
    c1 = u[:, 2] 
    r1 = u @ w @ v
    
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1
    c1 = c1.reshape((3,1))
    
    # 2nd Solution
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2 
    c2 = c2.reshape((3,1))
    
    # 3rd Solution
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3 
    c3 = c3.reshape((3,1)) 
    
    # 4th Solution
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c4 = c4.reshape((3,1))
    
    return [r1, r2, r3, r4], [c1, c2, c3, c4]
