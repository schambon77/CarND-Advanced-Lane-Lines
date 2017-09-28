# Imports
# =======
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Camera calibration
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

# Apply combined threshold to image
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

# Apply perspective transform to image
def warp_image(img):
    src = np.float32(
        [[689, 450],
         [1109, 719],
         [187, 719],
         [591, 450]])
    dst = np.float32(
        [[950, 0],
         [950, img.shape[0]],
         [300, img.shape[0]],
         [300, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped

def find_lines(warped_bin, debug=0, debug_dir='./', img_file='img'):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_bin[np.int(warped_bin.shape[0]*3/4):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_bin, warped_bin, warped_bin)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_bin.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_bin.shape[0] - (window + 1) * window_height
        win_y_high = warped_bin.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_bin.shape[0] - 1, warped_bin.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    if debug == 1:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, out_img.shape[1])
        plt.ylim(out_img.shape[0], 0)
        plt.savefig(debug_dir + 'lines_bin_' + img_file)
        plt.clf()

    return left_fit, right_fit

# Process image pipeline
def process_image(img, mtx, dist, debug=0, debug_dir='./', img_file='img'):
    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Color and gradient threshold
    bin = threshold_image(img, debug=debug, debug_dir=debug_dir, img_file=img_file)
    if debug == 1:
        mpimg.imsave(debug_dir + 'thresh_' + img_file, bin, cmap='gray')

    # Perspective transform
    warped = warp_image(img)
    warped_bin = warp_image(bin)
    if debug == 1:
        mpimg.imsave(debug_dir + 'warped_' + img_file, warped)
        mpimg.imsave(debug_dir + 'warped_bin_' + img_file, warped_bin, cmap='gray')

    # Finding the lines
    find_lines(warped_bin, debug=debug, debug_dir=debug_dir, img_file=img_file)

    # Measure curvature

    # Display lane area

# Process test images
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
