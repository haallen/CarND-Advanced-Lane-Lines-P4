#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:29:04 2017

@author: hope


The goals / steps of this project are the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""
#%% initialization
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

#path to calibration file and images used/generated in calibration process
calPath = '/Users/hope/Documents/python/carND/CarND-Advanced-Lane-Lines/camera_cal/'
#name of camera calibration file
calFname = 'cameraCalibration.p'

#path to test images directory
testPath = '/Users/hope/Documents/python/carND/CarND-Advanced-Lane-Lines/test_images/'
#%%
#Step1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#Note: much of this has been leveraged from https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
#This only needs to be done once and the results saved off for further use
def calibrateCamera(calPath, calFname):
    nx = 9 #number of corners along x
    ny = 6 #number of corners along y
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob(calPath + 'calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        
        # convert img to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = calPath + 'corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
    
            cv2.imshow('img', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(calPath + calFname, "wb" ) )
    
    # Test undistortion on an image
    img = cv2.imread(calPath + 'calibration1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(calPath + 'test_undist_cal1.jpg',dst)
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)

#%%
#Step 2: Apply a distortion correction to raw images.
def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

#%% Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
def color_gradient_threshold(img):
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    f.tight_layout()
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)
    
    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return combined_binary

#%% Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
def perspective_transform(img):
    #assume camera orientation is fixed 
    
    #source points
    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    
    #destination points
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp(img, M):
    
    img_size = (img.shape[1], img.shape[0])
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img,cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(warped,cmap='gray')
    ax2.set_title('Undistorted and Warped Image')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Return the resulting image and matrix
    return warped

#%% Step 5: Detect lane pixels and fit to find the lane boundary.
def detectNewLaneLines(img):
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Set the width of the windows +/- margin
    margin = 100
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))*255
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[np.int(img.shape[0]/2):,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
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
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.figure()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    return left_fit, right_fit, lefty, leftx, righty, rightx, left_fitx, right_fitx

def detectExistingLaneLines(img, left_fit, right_fit):
    # Assume you now have a new warped image from the next frame of video 

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin = 100
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    return left_fit, right_fit, lefty, leftx, righty, rightx, left_fitx, right_fitx
#%%Step 6: Determine the curvature of the lane and vehicle position with respect to center.
def calculateCurvature(lefty, leftx, righty, rightx):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_y_eval = np.max(lefty)
    right_y_eval =np.max(righty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix,  leftx*xm_per_pix, 2)
    right_fit_cr =np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0] *left_y_eval*ym_per_pix  + left_fit_cr[1])**2)**1.5)  / np.absolute(2*left_fit_cr[0])
    right_curverad =((1 + (2*right_fit_cr[0]*right_y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    
    return left_curverad, right_curverad

def calculatePosition(left_fitx, right_fitx):
    xm_per_pix = 3.7/700
    
    #center of image in pixels
    center = 1280.0/2
    
    #position from center of lane in meters
    pos_from_center = (center - (left_fitx[-1]+right_fitx[-1])/2)*xm_per_pix
    
    print(pos_from_center,'m')
    
    return pos_from_center

#%% Step 7: Warp the detected lane boundaries back onto the original image.

#%% Step 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# Create an image to draw the lines on
def drawing(undist, warped, left_fitx, right_fitx, leftCurve, rightCurve, offset):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, len(left_fitx)-1, len(left_fitx))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, Minv)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.figure()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.text(0,50,'Left curve: %.0f m; Right curve: %.0f m'%(leftCurve,rightCurve),color='white')
    plt.text(0,100,'Lane Offset: %.2f m'%offset, color='white')
    
    plt.imshow(result)
#%% main Pipeline

#check to see if camera has been calibrated, if not then calibrate camera
if os.path.isfile(calPath + calFname):
    #calDict is dictionary with mtx and dist as the keys
    calDict = pickle.load(open(calPath + calFname,'rb'))
else:
    calibrateCamera(calPath, calFname)
    #okay, i know this isn't very robust, but should be okay for now
    calDict = pickle.load(open(calPath + calFname,'rb'))

#read in test images
testImages = glob.glob(testPath + '*.jpg')
testImages = [testImages[0], testImages[1]]

M = None
left_fit = None
right_fit = None

for idx, fname in enumerate(testImages):
    #read in each image
    img = cv2.imread(fname)
    
    #undistort each image
    undistortedImg = undistort(img, calDict['mtx'], calDict['dist'])
    
    #apply color and gradient thresholding
    thresholdBinaryImg = color_gradient_threshold(undistortedImg)
    
    #calculate perspective transform matrix
    if M is None:
        M, Minv = perspective_transform(thresholdBinaryImg)
    
    #apply perspective transform
    birdsEye = warp(thresholdBinaryImg, M)
    
    #find lane lines
    if (left_fit is not None) and (right_fit is not None):
        left_fit, right_fit, lefty, leftx, righty, rightx, left_fitx, right_fitx = detectExistingLaneLines(birdsEye, left_fit, right_fit)
    else:
        left_fit, right_fit, lefty, leftx, righty, rightx, left_fitx, right_fitx = detectNewLaneLines(birdsEye)
    
    left_curverad, right_curverad = calculateCurvature(lefty, leftx, righty, rightx)
        
    center_of_lane = calculatePosition(left_fitx, right_fitx)
    
    rewarp = warp(birdsEye, Minv)
    
    drawing(undistortedImg, rewarp, left_fitx, right_fitx, left_curverad, right_curverad, center_of_lane)
        

