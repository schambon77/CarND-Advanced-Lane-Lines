# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration7.jpg "Calibration image"
[image2]: ./output_images/cal_image_with_corners.jpeg "Calibration image with corners drawn"
[image3]: ./test_images/test5.jpg "Test image"
[image4]: ./output_images/undistorted_image.jpeg "Undistorted test image"
[image5]: ./output_images/thresh_test5.jpg "Threholded image"
[image6]: ./output_images/warped_bin_test5.jpg "Warped binary image"
[image7]: ./output_images/warped_test5.jpg "Warped image"
[image8]: ./output_images/lines_bin_test5.jpg "Detected lines"
[image9]: ./output_images/final_test5.jpg "Image with lane and text info"
[video1]: ./output_video/lane_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

The source code can is contained in the file [advanced_lanes.py](https://github.com/schambon77/CarND-Advanced-Lane-Lines/blob/master/advanced_lanes.py)

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `get_calibration_matrix_and_distortion_coefs()`.

I first read in all the calibration images found in the camera_cal folder. Here is an example where the corners:

![alt text][image1]

I then iterate on each image to convert them to gray scale, find the chess board corners with the `cv2.findChessboardCorners()` function and append the returned image points to a list. Here are the detected corners from the previous calibration image:

![alt text][image2]

I record as well the object points corresponding to these corners, as the points of a 9x6 grid. I then apply the `cv2.calibrateCamera()` function to get the calibration matrix and distortion coefficients necessary to later on undistort images.

### Pipeline (single image)

The source code for the pipeline of the treatment of a single image is contained in the function `process_image()`. The function  `get_calibration_matrix_and_distortion_coefs()` is called only once before this function.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

 The calibration matrix and distortion coefficients, passed as arguments to `process_image()` are used with the function  `cv2.undistort()` in order to undistort images. The result is shown here:
 
![alt text][image4]

We can for instance notice that some details have disappeared at the edge of the image due to the undistortion transformation.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the `threshold_image()` function, I used a combination of different thresholding techniques in order to produce a binary image where the lane lines would stand out. The threshold methods used are:
* a threshold between 170 and 255 on the S channel after conversion to HLS color space (`hls_select()`)
* a gradient threshold (Sobel) in x direction between 20 and 100 (`abs_sobel_thresh()`)
* a gradient magniture threshold between 30 and 100 (`mag_thresh()`)
* a direction threshold between 0.7 and 1.3 rad (`dir_threshold()`)

In the combination, I keep pixels both detected by the magnitude and direction thresholds, and use an OR with the thresholded S channel and the x-oriented gradient. Here's the result on the test image:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform involves 2 functions:
* the `get_transform_matrix()` function is called outside the `process_image()` pipeline function. It uses the source and destination points shown below to generate through the `cv2.getPerspectiveTransform()` function the direct and inverse transformation matrices.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 689, 450      | 950, 0        | 
| 1109, 719      | 950, 719      |
| 187, 719     | 300, 719      |
| 591, 450      | 300, 0        |

* the `warp_image()` function is called within the single image pipeline. It applies the `cv2.warpPerspective()` function on the binary thresholded image using the transformation matrix obtained as indicated above. The result displayed below shows that the lines appear somewhat parallel in the warped image:

![alt text][image6]

As an additional check, I also apply the `warp_image()` on the test image:

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code used to detect the the lines is contained in the `find_lines()` function. It takes the binary warped image as an input, and then applied the first method presented in the course, with a few changes:
* computes an histogram on the bottom quarter of the image, to detect the 2 peaks representing the start of the left and right lines; the change from bottom half to bottom quarter was necessary to avoid too much noise towards the middle of the image.
* iterates upwards over 7 segments to find all potential line pixels, and adjust the center of the lines
* once all line pixels are supposed to be found, fits a 2nd-order polynomial for each line
* and performs the same fitting in 'world space' coordinates
Both types of fit are returned as they will be used for later on for different purposes.
The plotted fit results are shown on the test image here:

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The `measure_curvature()` function contains the code to compute the radius of curvature of the lane. I apply the formula presented in the course to both lines fit in world space at the bottom of the image (closest to the car). The 2 curvatures are returned, although the average between the 2 will eventually be displayed on the image (see below).

The `compute_distance_to_center()` function contains code to compute the position of the car with respect to the lane. I simply apply the lines fit (in pixel space) at the bottom of the image (closest to the car), which give me the precise point where the line starts in the picture. From this I can easily compute distance to the center of the lane (middle between lines) and I convert the distance to world space to represent a realistic value. The function returns the distance to the center as well as the lane width (useful for sanity checks).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `display_lane_area()` function is used to plot the lane back on the undistorted color image. It uses both left and right fits to compute the entire lines across the image (for all x), which is drawn on a blank image along the pixels in between lines. The function then uses the inverse warping transformation matrix in the `cv2.warpPerspective()` to warp the lane image and project t with some transparency on top of the undistorted color image.

The `display_text_info()` function is used to add textual information with regards to lane curvature and distance to center on the top of the image.

The resulting picture can be seen here:

![alt text][image9]

### Pipeline (video)

For the pipeline on the video, I have implemented the class `Line()` to track the state of the left and right lines detection. Once the `find_lines()` returns a fit, both line instances are marked detected, and the current fits are updated. For the next image frame, the  function `find_lines_nearby()` is called. It uses the fits (in pixel space) as a way to quickly identify potential line pixels, within a margin of 100 pixels left and right. Identified pixels are then used to refresh the line fits, in both pixel and world space.

Simple sanity checks are used to confirm the fits are still valid:
* the lane width has to be between 2 and 5 m
* the left and right line curvature have to be above 100m
If the checks pass, the current fit is validated and used with the last 10 (if available) to compute an average best fit.
It the check fail, the current fit is discarded and the previous one (if available) is used. After 10 failures to pass fit checks, the lines are marked not detected, which triggers a full detection process through `find_lines()` at the next frame.

Note: more stringent checks should be implemented for the more challenging videos.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/lane_project_video.mp4)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

In this challenging project, I was glad to take a step by step approach, with additional debugging code to save intermediate images in order to assess the effect of each transform. Although I have extensively reused techniques presented in the course material, this clearly helped me to implement and test the pipeline.

The selection of warping transform points was important, and took me some time to get right.

As can be seen in the video, the main current issue comes from the transition with shadows on the road. I think more targetted focus on this particular kind of pictures, at the thresholding phase, would help, with different color spaces for instance.
