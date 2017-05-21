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

#%% Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").

#%% main Pipeline

#check to see if camera has been calibrated, if not then calibrate camera
if os.path.isfile(calPath + calFname):
    #calDict is dictionary with mtx and dist as the keys
    calDict = pickle.load(open(calPath + calFname,'rb'))
else:
    calibrateCamera(calPath, calFname)
    #okay, i know this isn't very robust, but should be okay for now
    calDict = pickle.load(open(calPath + calFname,'rb'))

testImages = glob.glob(testPath + '*.jpg')
testImages = [testImages[0]]

for idx, fname in enumerate(testImages):
    img = cv2.imread(fname)
    undistortedImg = undistort(img, calDict['mtx'], calDict['dist'])
    cv2.imshow('image',undistortedImg)
    
    
        

