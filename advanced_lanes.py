# Imports
# =======
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Camera calibration
# ==================

def get_calibration_matrix_and_distortion_coefs():
    images = glob.glob('./camera_cal/calibration*.jpg')

    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    image_shape = 0
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    return mtx, dist

mtx, dist = get_calibration_matrix_and_distortion_coefs()

# Read test images
# ================

test_files = os.listdir('./test_images')
test_images = []
for test_file in test_files:
    img = mpimg.imread('./test_images/' + test_file)
    test_images.append(img)


# Distortion correction
# =====================

test_undists = []
for test_image in test_images:
    img = cv2.undistort(test_image, mtx, dist, None, mtx)
    test_undists.append(img)

# Color and gradient threshold

# Perspective transform

# Finding the lines

# Measure curvature