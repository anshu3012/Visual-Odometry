import random
import numpy as np
import cv2


def checkFmatrix(x1, x2, F):
    x11 = np.array([x1[0], x1[1], 1]).T
    x22 = np.array([x2[0], x2[1], 1])
    return abs(np.squeeze(np.matmul((np.matmul(x22, F)), x11)))


############################################################################################
#Use with caution 
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # descriptors from sift

features1 = []  # Variable for storing all the required features from the current frame
features2 = []  # Variable for storing all the required features from the next frame

# Ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        features1.append(keypoints1[m.queryIdx].pt)
        features2.append(keypoints2[m.trainIdx].pt)


noOfInliers = 0
finalFundMatrix = np.zeros((3, 3))

inlier1 = []  # Variable for storing all the inliers features from the current frame
inlier2 = []  # Variable for storing all the inliers features from the next frame

############################################################################################


for i in range(0, 50):  # 50 iterations for RANSAC
    count = 0
    eightpoint = []
    goodFeatures1 = []  # Variable for storing eight random points from the current frame
    goodFeatures2 = []  # Variable for storing corresponding eight random points from the next frame
    tempfeature1 = []
    tempfeature2 = []

    while(True):  # Loop runs while we do not get eight distinct random points
        num = random.randint(0, len(features1) - 1)
        if num not in eightpoint:
            eightpoint.append(num)
        if len(eightpoint) == 8:
            break

    for point in eightpoint:  # Looping over eight random points
        goodFeatures1.append([features1[point][0], features1[point][1]])
        goodFeatures2.append([features2[point][0], features2[point][1]])

    # Computing Fundamentals Matrix from current frame to next frame
    FundMatrix = fundamentalMatrix(goodFeatures1, goodFeatures2)

    for number in range(0, len(features1)):

        # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
        if checkFmatrix(features1[number], features2[number], FundMatrix) < 0.01:
            count = count + 1
            tempfeature1.append(features1[number])
            tempfeature2.append(features2[number])

    if count > noOfInliers:
        noOfInliers = count
        finalFundMatrix = FundMatrix
        inlier1 = tempfeature1
        inlier2 = tempfeature2
