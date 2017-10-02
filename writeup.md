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
[image6]: ./output_images/warped_bin_test5.jpg "Warped image"
[video1]: ./project_video.mp4 "Video"

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
