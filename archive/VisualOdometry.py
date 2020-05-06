'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       VisualOdometry.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


'''
@brief      
'''
class VisualOdometry:
    

    def __init__(self):
        pass
    

    '''
    @brief      Find matching features between two frames
    @param      
    @return       
    '''
    def findFeatureMatches(self, image1, image2):
        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6,        # 12
                           key_size = 12,           # 20
                           multi_probe_level = 1)   # 2
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0,0] for i in range(len(matches))]

        # Store all the good matches as per Lowe's ratio test.
        good = []
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
                matchesMask[i]=[1,0]

        pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        '''
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

        img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)

        cv2.imshow('frame', cv2.resize(img3, (1280, 720)))
        cv2.waitKey(0)
        '''

        return pts1, pts2


    '''
    @brief      Estimate Fundamental matrix from feature matches
    @param      
    @return       
    '''
    def getFundamentalMatrix(self, pts1, pts2):
        numPoints = pts1.shape[0]
        A = np.empty((numPoints, 9))

        for i in range(numPoints): 
            x1 = pts1[i,0]
            y1 = pts1[i,1]
            x2 = pts2[i,0]
            y2 = pts2[i,1]
            A[i] = np.array([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])
            #A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)     # Take SVD of A
        f = VT[-1].reshape((3,3))                             # Last column of V matrix
        
        # Constrain Fundamental Matrix to Rank 2
        u1,s1,v1 = np.linalg.svd(f) 
        s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) 
        F = u1 @ s2 @ v1

        return F
    
    
    '''
    @brief      Reject outliers from a set of feature matches using RANSAC
    @param      
    @return       
    '''
    def rejectFeatureMatchOutliers(self, pts1, pts2, iterations=100, epsilon=0.001):
        # Variable to keep track of the max number of inliers so far
        numInliers = 0

        # Stores indices of inliers
        inlierIndices = []

        for i in range(0, iterations): 
            # Inlier count for this iteration
            count = 0

            # Inlier indices for this iteration
            tempInlierIndices = []

            # Set of 8 feature matches to construct F matrix
            modelPoints1 = []
            modelPoints2 = []
            
            # Generate 8 random indices
            randomIndices = random.sample(range(0, pts1.shape[0]), 8)

            for index in randomIndices:
                modelPoints1.append(pts1[index])
                modelPoints2.append(pts2[index])

            # Compute Fundamental Matrix from model points
            F = self.getFundamentalMatrix(np.array(modelPoints1), np.array(modelPoints2))

            for j in range(pts1.shape[0]):
                x1 = np.array([[pts1[j,0]], [pts1[j,1]], [1]])
                x2 = np.array([[pts2[j,0]], [pts2[j,1]], [1]])
                error = np.linalg.norm(x2.T @ F @ x1)

                # If x2.T * F * x1 is less than epsilon then count the match as an inlier
                if (error < epsilon):
                    count += 1
                    tempInlierIndices.append(j)
            
            # Update numInliers and inlierIndices if necessary
            if (count > numInliers):
                numInliers = count
                inlierIndices = tempInlierIndices

        # Extract inliers from pts1 and pts2
        inliers1 = np.array([pts1[i] for i in inlierIndices])
        inliers2 = np.array([pts2[i] for i in inlierIndices])

        return inliers1, inliers2



    '''
    @brief      Estimate Essential matrix from Fundamental matrix F by accounting for camera calibration parameters
    @param      
    @return       
    '''
    def getEssentialMatrix(self, F, K):
        e = K.T @ F @ K

        # Constrain Essential Matrix to have specific singular values
        U,S,VT = np.linalg.svd(e) 
        S2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) 
        E = U @ S2 @ VT

        return E

    
    def skew(self, v):
        a = v[0]
        b = v[1]
        c = v[2]
        return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


    def linear_triangulation(K, P0, P1, pt1, pt2):
        pt1 = np.insert(np.float32(pt1), 2, 1)
        pt2 = np.insert(np.float32(pt2), 2, 1)

        homo_pt1 = np.matmul(np.linalg.inv(K), pt1.reshape((-1, 1)))
        homo_pt2 = np.matmul(np.linalg.inv(K), pt2.reshape((-1, 1)))

        skew0 = skew(homo_pt1.reshape((-1,)))
        skew1 = skew(homo_pt2.reshape((-1,)))

        P0 = np.concatenate((P0[:, :3], -np.matmul(P0[:, :3], P0[:, 3].reshape(-1, 1))), axis=1)
        P1 = np.concatenate((P1[:, :3], -np.matmul(P1[:, :3], P1[:, 3].reshape(-1, 1))), axis=1)
        
        pose1 = np.matmul(skew0, P0[:3, :])
        pose2 = np.matmul(skew1, P1[:3, :])

        # Solve the equation Ax=0
        A = np.concatenate((pose1, pose2), axis=0)
        u, s, vt = np.linalg.svd(A)
        X = vt[-1]
        X = X / X[3]

        return X


    '''
    @brief      Find correct translation/rotation vectors from Essential matrix from depth positivity
    @param      
    @return       
    '''
    def findPose(self):
        pass


    '''
    @brief      Run VO pipeline on image stream and plot camera pose
    @param      
    @return       
    '''
    def runApplication(self):
        # Create video stream object
        videoCapture = cv2.VideoCapture(videoFile)
        
        # Define video codec and output file if video needs to be saved
        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 720p 30fps video
            out = cv2.VideoWriter('BuoyDetection.mp4', fourcc, 30, (1280, 720))

        # Continue to process frames if the video stream object is open
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            # Continue processing if a valid frame is received
            if ret == True:
                newFrame = self.detectBuoys(frame)

                # Save video if desired, resizing frame to 720p
                if (saveVideo == True):
                    out.write(cv2.resize(newFrame, (1280, 720)))
                
                # Display frame to the screen in a video preview
                cv2.imshow("Frame", cv2.resize(newFrame, (1280, 720)))

                # Exit if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # If the end of the video is reached, wait for final user keypress and exit
            else:
                cv2.waitKey(0)
                break
        
        # Release video and file object handles
        videoCapture.release()
        if (saveVideo == True):
            out.release()
        
        print('Video and file handles closed')
        



if __name__ == '__main__':
    K = np.array([[964.828979,  0,          643.788025],
                  [0,           964.828979, 484.40799 ],
                  [0,           0,          1         ]])
    # Read test images
    image1 = cv2.imread('test1.png', 0)       
    image2 = cv2.imread('test2.png', 0) 

    vOdom = VisualOdometry()

    pts1, pts2 = vOdom.findFeatureMatches(image1, image2)

    in1, in2 = vOdom.rejectFeatureMatchOutliers(pts1, pts2)

    F = None
    if (in1.shape[0] >= 8):
        F = vOdom.getFundamentalMatrix(in1, in2)

    E = vOdom.getEssentialMatrix(F, K)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    U, S, VT = np.linalg.svd(E) 

    # Get the four possible poses
    C1 = U(:,3)
    R1 = U @ W @ VT

    C2 = −U(:,3)
    R2 = U @ W @ VT

    C3 = U(:,3)
    R3 = U @ W.T @ VT

    C4 = −U(:,3)
    R4 = U @ W.T @ VT

    # Correct poses based on determinant if necessary
    if (np.linalg.det(R1) < 0):
        R1 = -R1
        C1 = -C1
    if (np.linalg.det(R2) < 0):
        R2 = -R2
        C2 = -C2
    if (np.linalg.det(R3) < 0):
        R3 = -R3
        C3 = -C3
    if (np.linalg.det(R4) < 0):
        R4 = -R4
        C4 = -C4
    
    P1 = np.concatenate((R1, C1), axis=1)
    P2 = np.concatenate((R2, C2), axis=1)
    P3 = np.concatenate((R3, C3), axis=1)
    P4 = np.concatenate((R4, C4), axis=1)
    
    for i in range(in1.shape[0]):
        

    

