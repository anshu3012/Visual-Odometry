import numpy as np
import cv2
import os

from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

# Image and model directories
imageDirectory = '../Oxford_dataset/stereo/centre/'
modelDirectory = '../Oxford_dataset/model'

images = []

# Read and save camera intrinsic parameters
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(modelDirectory)
'''
K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]]) 
f = open('Camera Intrinsics.txt','a')
f.write('K: ' + '\n' + str(K) + '\n\n')
f.write('G_camera_image: ' + '\n' + str(G_camera_image))
f.close()
'''

# Process all images in the image directory
for imageName in os.listdir(imageDirectory):
    rawImage = cv2.imread(imageDirectory + imageName)

    # Convert raw image to grayscale and demosaic into BGR image
    gray = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    imageBGR = cv2.cvtColor(gray, cv2.COLOR_BayerGR2BGR)

    # Undistort BGR image
    undistoredImage = UndistortImage(imageBGR, LUT)

    # Save to dataset folder
    cv2.imwrite('dataset/' + imageName, undistoredImage)




