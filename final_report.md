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
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
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

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The identification of lane line pixels and fitting to a polynomial happens in lines #220 through #378 of [file](AdvancedLaneDetection_video.py). The input is the warped, thresholded image. I have two separate functions, detectNewLaneLines and detectExistingLaneLines, that are called depending on whether to leverage previous lane fit information or to start from scratch. The decision on whether or not to start from scratch will be detailed later. 

detectNewLaneLines - Creates histogram of the bottom half of the image and finds the peak of the left and right hand side of the histogram. These position of these two peaks will be the starting points for the left and right lane line pixel positions, respectively. The image is then divided into a specified number of smaller windows starting at the x positions of the histogram peaks. Each window is iterated over and the indices of the nonzero pixels within each window are identified and stored. With each iteration, the position of the window is determined based on the average x-positions of the previous window's detected nonzero pixels. When all of the windows have been processed, two lines are fit with a second order polynomial through the non-zero pixel positions in all of the windows.

detectExistingLaneLines - leverages the current lane line fit. Does not require histograming or the sliding window approach of detectNewLaneLines. Instead, it looks for nonzero pixels along the line. 

In my pipeline, I have two checks for whether a new lane line fit should be calculated or to use the current fit. The first check is a check to see if any pixels were detected along the current line. If not, then a new fit should be calculated. The second check is a very coarse sanity check. It compares the calculated curvature of each line, the slopes of each line, and the distance in pixels between each line. If any of these values are way off (like the wrong order of magnitude), then it says that the lines for that image are not valid and that the previous line should be drawn on the image. I do have a counter that waits for N number of bad frames before refitting the line. Update - I removed the sanity checks on curvature and slope; they proved to be very noisy and I would like to investigate further before adding them back in.

Here's an example of my polynomial fit through non-zero pixels: 

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

If I had more time I would investigate two items. The first would be to fine-tune my color and gradient threshold approach. I am interested in leveraging the other color channels as the gradient direction information. The threshold values that I used could also stand to be optimized.

The second item that I would look into is smoothing my lane lines across frames of the video. I am using the Lines class as suggested in the lectures, but haven't had time to determine the best approach to averaging across frames. 

I can also imagine that my implementation could have issues if there is a white or yellow car directly in front of it in the same lane. I would also like to investigate different lighting conditions. The provided video and test images were taken in bright daylight and I could imagine my color and gradient thresholding having issues in other types of lighting. 
