from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os



images = []
path = "stereo/centre/" 
for image in os.listdir(path): # Looping over all the images
    images.append(image) # Storing all the image names in a list
    images.sort() # Sorting the image names

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]]) # Camera Calibration Matrix of the model
