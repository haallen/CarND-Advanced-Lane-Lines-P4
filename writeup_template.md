**Advanced Lane Finding Project**

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

[image1]: ./output_images/calibration.png "Distorted and Undistorted Comparison of Calibration Image 1"
[image2]: ./output_images/undistort_example.jpg "Undistorted Test Image"
[image3]: ./output_images/color_gradient_threshold.png "Color and Gradient Thresholding"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained lines 32-99 of this [file](AdvancedLaneDetection_video.py).  

Note - Some of this section has been copied from provided template; I thought it was a pretty good starting point

I start by preparing objp, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. 

imgpoints is a list that will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

For each of the provided calibration images:
 - convert to greyscale
 - use cv2's findChessboardCorners method to find the inside corners of the image
 - append objp to the objpoints list
 - append the detected corners to imgpoints list
 - draw the detected corners on the image (not required but a good sanity check)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The camera calibration and distortion coefficients were saved to a file for use in my pipeline.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)
This section describes my basic pipeline and provides examples for each step. The same basic pipeline was used to process both the provided test images and the videos. 

#### 1. Provide an example of a distortion-corrected image.

The camera matrix and distortion coefficients that were calculated as described above are applied to each image. Here's an example of an undistorted test image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #110 through #167 in [file](AdvancedLaneDetection_video.py).  I implemented the algorithms described in the 'Color and Gradient' lesson to use both directional gradient and color thresholding. 

I first apply direction gradient thresholding by converting the image to greyscale and use the Sobel operator to take the gradient in the x direction, take the absolute value of the gradient, and then scale the result to a value between 0 and 255. Then a binary threshold was applied. 

I then performed color thresholding by converting the image into HLS color space and thresholding on both the saturation and the brightness channels. By combining the outputs of these thresholds, I am making sure that yellow and white lines are detected and shadows and other dark lines are ignored. 

I arrived at the values for all my thresholds through trial and error. More time could be spent here to find optimal thresholds.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in lines 192 through 218 in the file [file](AdvancedLaneDetection_video.py) (output_images/examples/example.py) .

The perspective_transform function calculates the perspective transform matrix and inverse for a hardcoded set of source and destination points
I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 350, 720      |
| 575, 475      | 350, 0        | 
| 720, 475      | 975, 0        |
| 1125, 720     | 975, 720      |

The perspective matrix returned by the perspective_transform method is input into the warm method along with the thresholded image. The matrix is applied to the image and returned back to the pipeline.  

Here's an example of the perspective transformation:
![alt text][image4]

Note: before I applied the perspective transform, I applied a mask to the output of the thresholding so that only the pixels in a region of interest are processed. 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The identification of lane line pixels and fitting to a polynomial happens in lines #220 through #378 of [file](AdvancedLaneDetection_video.py). The input is the warped, thresholded image. I have two separate functions, detectNewLaneLines and detectExistingLaneLines, that are called depending on whether to leverage previous lane fit information or to start from scratch. The decision on whether or not to start from scratch will be detailed later. 

detectNewLaneLines - Creates histogram of the bottom half of the image and finds the peak of the left and right hand side of the histogram. These position of these two peaks will be the starting points for the left and right lane line pixel positions, respectively. The image is then divided into a specified number of smaller windows starting at the x positions of the histogram peaks. Each window is iterated over and the indices of the nonzero pixels within each window are identified and stored. With each iteration, the position of the window is determined based on the average x-positions of the previous window's detected nonzero pixels. When all of the windows have been processed, two lines are fit with a second order polynomial through the non-zero pixel positions in all of the windows.

detectExistingLaneLines - leverages the current lane line fit. Does not require histograming or the sliding window approach of detectNewLaneLines. Instead, it looks for nonzero pixels along the line. 

In my pipeline, I have two checks for whether a new lane line fit should be calculated or to use the current fit. The first check is a check to see if any pixels were detected along the current line. If not, then a new fit should be calculated. The second check is a very coarse sanity check. It compares the calculated curvature of each line, the slopes of each line, and the distance in pixels between each line. If any of these values are way off (like the wrong order of magnitude), then it says that the lines for that image are not valid and that the previous line should be drawn on the image. I do have a counter that waits for N number of bad frames before refitting the line. Update - I removed the sanity checks on curvature and slope; they proved to be very noisy and I would like to investigate further before adding them back in.

Here's an example of my polynomial fit through non-zero pixels: 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #381 through #415 in my code.

To calculate the curvature, I scaled the lane-line pixel positions for both the left and right lane lines to meters using the provided conversion factor. I then fit lines through the scaled points and calculated the radius of curvature for each of the lines using the provided formula. 

To calculate position of vehicle with respect to center, I used the fitted parameters for the left and right lane lines to calculate the x pixel value where each lane line crosses the x-axis. I averaged the left lane and right lane x values and subtracted the average from the value of the center of the image. I then converted this distace to meters to obtain the position of the vehicle with respect to the center of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #422 through #454 in my code. Not much to say here.  

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If I had more time I would investigate two items. The first would be to fine-tune my color and gradient threshold approach. I am interested in leveraging the other color channels as the gradient direction information. The threshold values that I used could also stand to be optimized.

The second item that I would look into is smoothing my lane lines across frames of the video. I am using the Lines class as suggested in the lectures, but haven't had time to determine the best approach to averaging across frames. 

I can also imagine that my implementation could have issues if there is a white or yellow car directly in front of it in the same lane. I would also like to investigate different lighting conditions. The provided video and test images were taken in bright daylight and I could imagine my color and gradient thresholding having issues in other types of lighting. 


