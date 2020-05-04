'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       VisualOdometry.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

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
    def findPointCorrespondences(self, image1, image2):
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
    @brief      Estimate Fundamental matrix from point correspondences using RANSAC
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
            A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y2*y1, y1, x2, y2, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)     # Take SVD of A
        f = VT[-1].reshape(3,3)                             # Last column of V matrix
        
        # Constrain Fundamental Matrix to Rank 2
        u1,s1,v1 = np.linalg.svd(f) 
        s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) 
        F = u1 @ s2 @ v1

        return F
    
    
    '''
    @brief      Estimate Essential matrix from Fundamental matrix F by accounting for camera calibration parameters
    @param      
    @return       
    '''
    def getEssentialMatrix(self):
        pass

    
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
    def runApplication(self, videoFile, saveVideo=False):
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
    # Read test images
    image1 = cv2.imread('test1.png', 0)       
    image2 = cv2.imread('test2.png', 0) 

    vOdom = VisualOdometry()

    pts1, pts2 = vOdom.findPointCorrespondences(image1, image2)

    F0 = vOdom.getFundamentalMatrix(pts1, pts2)
    print(F0)

    F,_ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    print(F)

