'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       VisualOdometry.py
@date       2020/04/02
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2


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
    def findPointCorrespondences(self):
        pass


    '''
    @brief      Estimate Fundamental matrix from point correspondences using RANSAC
    @param      
    @return       
    '''
    def getFundamentalMatrix(self):
        pass
    
    
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
    pass
