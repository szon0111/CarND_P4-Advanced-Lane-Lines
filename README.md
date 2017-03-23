# Project 4: Advanced Lane Finding (in progress)

[//]: # (Image References)

[image1]: https://cloud.githubusercontent.com/assets/10526591/24232250/f4afdd68-0fcc-11e7-8f43-410fa73e7b97.jpg "Video Thumbnail"

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*please see [writeup.md](https://github.com/szon0111/CarND_P4-Advanced-Lane-Lines/blob/master/writeup.md) for a detailed report*

Overview
---
Detect lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting. The software pipeline will identify the lane boundaries in a video from a front-facing camera on a car. 

#### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Project Deliverables
---
* `calibrate.py` to calibrate the camera using provided chessboard images.
* `undistort.py` to undistort test images using the result from `calibrate.py`.
* `threshold.py` to apply various threshold methods and return thresholded binary output image.
* `perspective.py` to transfrom perspective of the image to bird's eye view.
* `polyfit.py` to detect lane lines, calculate curvature and calculate vehicle position from center of the lane.
* `polydraw.py` to draw the detected lane markings and transform the image back to original image space.
* `video.py` to produce the video with lane markings, curvature value and vehicle position value.
* `project_video_output.mp4` is the final ouput video

Results
---
View the **[video](https://youtu.be/E1UGFOa2ado)** on Youtube

![Video][image1]
