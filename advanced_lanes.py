# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Constants
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/650 # meters per pixel in x dimension

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
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped, M, Minv

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
    nwindows = 7
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

    # Do the same in world space
    left_fit_w = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_w = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

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

    return left_fit, right_fit, left_fit_w, right_fit_w

def measure_curvature(y_eval, left_fit_w, right_fit_w):
    left_curverad = ((1 + (2 * left_fit_w[0] * y_eval * ym_per_pix + left_fit_w[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_w[0])
    right_curverad = ((1 + (2 * right_fit_w[0] * y_eval * ym_per_pix + right_fit_w[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_w[0])
    return left_curverad, right_curverad

def compute_distance_to_center(img_shape, left_fit, right_fit, debug=0):
    left_lane_x = left_fit[0] * img_shape[0] ** 2 + left_fit[1] * img_shape[0] + left_fit[2]
    right_lane_x = right_fit[0] * img_shape[0] ** 2 + right_fit[1] * img_shape[0] + right_fit[2]
    dist_to_center = (img_shape[1]/2) - ((right_lane_x + left_lane_x)/2)
    dist_to_center_w = dist_to_center * xm_per_pix
    if debug == 1:
        print('Left lane: {}'.format(left_lane_x))
        print('Right lane: {}'.format(right_lane_x))
        print('Distance to center: {}'.format(dist_to_center))
        print('Distance to center: {:.2f}m'.format(dist_to_center_w))
        print('')
    return dist_to_center_w

def display_lane_area(warped, undist, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0]-1, num=warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def display_text_info(img, curverad, distance_to_center_w):
    cv2.putText(img, 'Radius of curvature: {:7.0f}m'.format(curverad), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    right_left = 'left'
    if distance_to_center_w >= 0:
        right_left = 'right'
    cv2.putText(img, 'Vehicle is {:5.2f}m {} of center'.format(distance_to_center_w, right_left), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    return img

# Process image pipeline
def process_image(img, mtx, dist, debug=0, debug_dir='./', img_file='img'):
    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if debug == 1:
        mpimg.imsave(debug_dir + 'undist_' + img_file, undist)

    # Color and gradient threshold
    bin = threshold_image(undist, debug=debug, debug_dir=debug_dir, img_file=img_file)
    if debug == 1:
        mpimg.imsave(debug_dir + 'thresh_' + img_file, bin, cmap='gray')

    # Perspective transform
    warped_bin, M, Minv = warp_image(bin)
    if debug == 1:
        warped, M, Minv = warp_image(img)
        mpimg.imsave(debug_dir + 'warped_' + img_file, warped)
        mpimg.imsave(debug_dir + 'warped_bin_' + img_file, warped_bin, cmap='gray')

    # Finding the lines
    left_fit, right_fit, left_fit_w, right_fit_w = find_lines(warped_bin, debug=debug, debug_dir=debug_dir, img_file=img_file)

    # Measure curvature
    left_curverad, right_curverad = measure_curvature(img.shape[0], left_fit_w, right_fit_w)
    if debug == 1:
        print('Image: {}'.format(img_file))
        print('Left curvature radius: {:.1f}m'.format(left_curverad))
        print('Right curvature radius: {:.1f}m'.format(right_curverad))
        print('Average curvature radius: {:.1f}m'.format((left_curverad + right_curverad)/2))
        print('')

    # Compute distance from center of lane
    distance_to_center_w = compute_distance_to_center(img.shape, left_fit, right_fit, debug=debug)

    # Display lane area
    undist_with_lane = display_lane_area(warped_bin, undist, left_fit, right_fit, Minv)
    if debug == 1:
        mpimg.imsave(debug_dir + 'undist_with_lane_' + img_file, undist_with_lane)

    # Display text info
    final = display_text_info(undist_with_lane, (left_curverad + right_curverad)/2, distance_to_center_w)
    if debug == 1:
        mpimg.imsave(debug_dir + 'final_' + img_file, final)

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
