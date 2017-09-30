# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import collections

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

# Get perspective transform matrices
def get_transform_matrix():
    src = np.float32(
        [[689, 450],
         [1109, 719],
         [187, 719],
         [591, 450]])
    dst = np.float32(
        [[950, 0],
         [950, 719],
         [300, 719],
         [300, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

# Apply perspective transform to image
def warp_image(img, M):
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

def find_lines_nearby(warped_bin, left_fit, right_fit):
    nonzero = warped_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                      & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                       & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)
    # Do the same in world space
    left_fit_new_w = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_new_w = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    return left_fit_new, right_fit_new, left_fit_new_w, right_fit_new_w

def measure_curvature(y_eval, left_fit_w, right_fit_w):
    left_curverad = ((1 + (2 * left_fit_w[0] * y_eval * ym_per_pix + left_fit_w[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_w[0])
    right_curverad = ((1 + (2 * right_fit_w[0] * y_eval * ym_per_pix + right_fit_w[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_w[0])
    return left_curverad, right_curverad

def compute_distance_to_center(img_shape, left_fit, right_fit, debug=0):
    left_line_x = left_fit[0] * img_shape[0] ** 2 + left_fit[1] * img_shape[0] + left_fit[2]
    right_line_x = right_fit[0] * img_shape[0] ** 2 + right_fit[1] * img_shape[0] + right_fit[2]
    dist_to_center = (img_shape[1]/2) - ((right_line_x + left_line_x)/2)
    dist_to_center_w = dist_to_center * xm_per_pix
    lane_width_w = (right_line_x - left_line_x) * xm_per_pix
    if debug == 1:
        print('Left lane: {}'.format(left_line_x))
        print('Right lane: {}'.format(right_line_x))
        print('Distance to center: {}'.format(dist_to_center))
        print('Distance to center: {:.2f}m'.format(dist_to_center_w))
        print('Lane width: {:.2f}m'.format(lane_width_w))
        print('')
    return dist_to_center_w, lane_width_w

def display_lane_area(undist, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    img_zero = np.zeros((undist.shape[0], undist.shape[1])).astype(np.uint8)
    color_warp = np.dstack((img_zero, img_zero, img_zero))

    ploty = np.linspace(0, undist.shape[0]-1, num=undist.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
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

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # number of successive most recent missed frames
        self.missed = 0
        # number of most recent fits used for smoothing
        self.n = 10
        # x values of the last n fits of the line
        #self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        #self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients averaged over the last n iterations in world space
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the most recent fit in world space
        self.current_fit_w = [np.array([False])]
        #polynomial coefficients for the n most recent fit
        self.recent_fits = collections.deque(maxlen=self.n)
        #polynomial coefficients for the n most recent fit in world space
        self.recent_fits_w = collections.deque(maxlen=self.n)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        #self.allx = None
        #y values for detected line pixels
        #self.ally = None

    def validate_current_fit(self):
        self.recent_fits.append(self.current_fit)
        self.recent_fits_w.append(self.current_fit_w)
        self.best_fit = np.mean(self.recent_fits, axis=0)
        self.best_fit_w = np.mean(self.recent_fits_w, axis=0)

    def increment_missed(self):
        self.missed += 1
        if np.any(self.best_fit == None):
            self.best_fit = self.current_fit
        if np.any(self.recent_fits_w == None)   :
            self.best_fit_w = self.current_fit_w
        if (self.missed == self.n):
            self.recent_fits.clear()
            self.recent_fits_w.clear()
            self.detected = False
            self.missed = 0

# Process image pipeline
def process_image(img, mtx, dist, M, Minv, left_line, right_line, debug=0, debug_dir='./', img_file='img'):
    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if debug == 1:
        mpimg.imsave(debug_dir + 'undist_' + img_file, undist)

    # Color and gradient threshold
    bin = threshold_image(undist, debug=debug, debug_dir=debug_dir, img_file=img_file)
    if debug == 1:
        mpimg.imsave(debug_dir + 'thresh_' + img_file, bin, cmap='gray')

    # Perspective transform
    warped_bin = warp_image(bin, M)
    if debug == 1:
        warped = warp_image(img, M)
        mpimg.imsave(debug_dir + 'warped_' + img_file, warped)
        mpimg.imsave(debug_dir + 'warped_bin_' + img_file, warped_bin, cmap='gray')

    # check if one or both lines have not been previously detected
    if not left_line.detected or not right_line.detected:

        # Finding the lines
        left_fit, right_fit, left_fit_w, right_fit_w = find_lines(warped_bin, debug=debug, debug_dir=debug_dir, img_file=img_file)

        # Update lines current fit
        left_line.detected = True
        right_line.detected = True
        left_line.current_fit = left_fit
        left_line.current_fit_w = left_fit_w
        right_line.current_fit = right_fit
        right_line.current_fit_w = right_fit_w
        print('Find lines')

    else:  #both lines were previously detected

        # Finding the lines nearby based on previous lines
        left_fit, right_fit, left_fit_w, right_fit_w = find_lines_nearby(warped_bin, left_line.best_fit, right_line.best_fit)

        # Update lines current fit
        left_line.current_fit = left_fit
        left_line.current_fit_w = left_fit_w
        right_line.current_fit = right_fit
        right_line.current_fit_w = right_fit_w

    # Measure curvature
    left_curverad, right_curverad = measure_curvature(img.shape[0], left_line.current_fit_w, right_line.current_fit_w)
    if debug == 1:
        print('Image: {}'.format(img_file))
        print('Left curvature radius: {:.1f}m'.format(left_curverad))
        print('Right curvature radius: {:.1f}m'.format(right_curverad))
        print('Average curvature radius: {:.1f}m'.format((left_curverad + right_curverad)/2))
        print('')

    # Compute distance from center of lane
    distance_to_center_w, lane_width_w = compute_distance_to_center(img.shape, left_line.current_fit, right_line.current_fit, debug=debug)

    # Sanity checks to validate current fit
    if ((lane_width_w > 2)
        and (lane_width_w < 5)
        and (left_curverad > 100)
        and (right_curverad > 100)):
        left_line.validate_current_fit()
        right_line.validate_current_fit()
    else:
        left_line.increment_missed()
        right_line.increment_missed()

    # Display lane area
    undist_with_lane = display_lane_area(undist, left_line.best_fit, right_line.best_fit, Minv)
    if debug == 1:
        mpimg.imsave(debug_dir + 'undist_with_lane_' + img_file, undist_with_lane)

    # Display text info
    final = display_text_info(undist_with_lane, (left_curverad + right_curverad)/2, distance_to_center_w)
    if debug == 1:
        mpimg.imsave(debug_dir + 'final_' + img_file, final)

    return final

# Process test images
def process_test_images(mtx, dist, M, Minv, debug=0, debug_dir='./'):
    test_files = os.listdir('./test_images')
    for test_file in test_files:
        left_line = Line()
        right_line = Line()
        img_file = './test_images/' + test_file
        img = mpimg.imread(img_file)
        process_image(img, mtx, dist, M, Minv, left_line, right_line, debug=debug, debug_dir=debug_dir, img_file=test_file)

# Process video image
def process_video_image(img):
    result = process_image(img, mtx, dist, M, Minv, left_line, right_line, debug=0, debug_dir='./output_video/')
    return result

# Main function for advanced lane detection
if __name__ == "__main__":
    mode = 'video'  #toggle here between test images and videos
    mtx, dist = get_calibration_matrix_and_distortion_coefs()
    M, Minv = get_transform_matrix()
    if mode == 'images':
        debug_dir = './test/'
        if not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)
        process_test_images(mtx, dist, M, Minv, debug=1, debug_dir=debug_dir)
    elif mode == 'video':
        debug_dir = './test_video/'
        if not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)
        left_line = Line()
        right_line = Line()
        output_dir = './output_video/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        video_file = 'project_video.mp4'
        #video_file = 'challenge_video.mp4'
        #video_file = 'harder_challenge_video.mp4'
        video_output = output_dir + 'lane_' + video_file
        clip1 = VideoFileClip(video_file)
        white_clip = clip1.fl_image(process_video_image)
        white_clip.write_videofile(video_output, audio=False)