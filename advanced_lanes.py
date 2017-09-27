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

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    sob = gray
    if orient == 'x':
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sob = np.absolute(sob)
    abs_sob_scaled = np.int8(255*abs_sob/(np.max(abs_sob)))
    binary_output = np.zeros(abs_sob_scaled.shape)
    binary_output[(abs_sob_scaled > thresh_min) & (abs_sob_scaled < thresh_max)] = 1
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sob = np.sqrt(np.square(sobx) + np.square(soby))
    sob_scaled = np.int8(255*sob/np.max(sob))
    binary_output = np.zeros(sob_scaled.shape)
    binary_output[(sob_scaled > mag_thresh[0]) & (sob_scaled < mag_thresh[1])] = 1
    return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobx = np.absolute(sobx)
    abs_soby = np.absolute(soby)
    dir_grad = np.arctan2(abs_soby, abs_sobx)
    binary_output = np.zeros(dir_grad.shape)
    binary_output[(dir_grad > thresh[0]) & (dir_grad < thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_output = np.zeros(s.shape)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

def threshold_image(img, debug=0, debug_dir='./', img_file='img'):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s_bin = hls_select(img, thresh=(170, 255))
    sx_bin = abs_sobel_thresh(gray, orient='x', thresh_min=20, thresh_max=100)
    mag_bin = mag_thresh(gray, sobel_kernel=3, mag_thresh=(30, 100))
    dir_bin = dir_threshold(gray, sobel_kernel=3, thresh=(0.7, 1.3))
    mag_dir_bin = np.zeros_like(gray)
    mag_dir_bin[(mag_bin == 1) & (dir_bin == 1)] = 1
    combined_bin = np.zeros_like(gray)
    combined_bin[(s_bin == 1) | (sx_bin == 1) | ((mag_bin == 1) & (dir_bin == 1))] = 1
    if debug == 1:
        mpimg.imsave(debug_dir + 'thresh_s_' + img_file, s_bin, cmap='gray')
        mpimg.imsave(debug_dir + 'thresh_sx_' + img_file, sx_bin, cmap='gray')
        mpimg.imsave(debug_dir + 'thresh_mag_' + img_file, mag_bin, cmap='gray')
        mpimg.imsave(debug_dir + 'thresh_dir_' + img_file, dir_bin, cmap='gray')
        mpimg.imsave(debug_dir + 'thresh_mag_dir_' + img_file, mag_dir_bin, cmap='gray')
    return combined_bin

# Process image pipeline
# ======================
def process_image(img, mtx, dist, debug=0, debug_dir='./', img_file='img'):
    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Color and gradient threshold
    bin = threshold_image(img, debug=debug, debug_dir=debug_dir, img_file=img_file)
    if debug == 1:
        mpimg.imsave(debug_dir + 'thresh_' + img_file, bin, cmap='gray')

    # Perspective transform

    # Finding the lines

    # Measure curvature

# Process test images
# ===================
def process_test_images(mtx, dist, debug=0, debug_dir='./'):
    test_files = os.listdir('./test_images')
    for test_file in test_files:
        img_file = './test_images/' + test_file
        img = mpimg.imread(img_file)
        process_image(img, mtx, dist, debug=debug, debug_dir=debug_dir, img_file=test_file)

debug_dir = './test/'
if not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)
mtx, dist = get_calibration_matrix_and_distortion_coefs()
process_test_images(mtx, dist, debug=1, debug_dir=debug_dir)
