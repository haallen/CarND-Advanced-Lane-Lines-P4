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

I first apply direction gradient thresholding by converting the image to greyscale and use the Sobel operator to take the gradient in the x direction, take the absolute value of the gradient, and then scale the result to a value between 0 and 255. Then a binary threshold was applied. I arrived at the values of 20 and 100 for my thresholds through trial and error. More time could be spent here to find optimal thresholds.

I then performed color thresholding by converting the image into HLS color space and thresholding on both the saturation and the brightness channels. By combining the outputs of these thresholds, I am making sure that yellow and white lines are detected and shadows and other dark lines are ignored. 

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in lines 192 through 218 in the file [file](AdvancedLaneDetection_video.py) (output_images/examples/example.py) .

The perspective_transform function calculates the perspective transform matrix and inverse for a hardcoded set of source and destination points
I chose the hardcode the source and destination points in the following manner:

```python
  #source points
    src = np.float32([[(200, 720), (575, 475), (720, 475), (1125, 720)]])
    
    #destination points
    dst = np.float32([[(350, 720), (350, 0), (975, 0), (975, 720)]])
```

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
