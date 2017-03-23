# **Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: https://cloud.githubusercontent.com/assets/10526591/24229793/c15ed472-0fbe-11e7-8087-513568deabaa.jpg "Original"
[image2]: https://cloud.githubusercontent.com/assets/10526591/24229828/f80b124c-0fbe-11e7-93b9-2e5d1a2492ba.jpg "Undistorted"
[image3]: https://cloud.githubusercontent.com/assets/10526591/24230407/51964d56-0fc2-11e7-889e-98671db07710.png "Binary Example"
[image4]: https://cloud.githubusercontent.com/assets/10526591/24230966/475fa2e4-0fc5-11e7-9b31-5c7d5d22f35c.jpg "Original"
[image5]: https://cloud.githubusercontent.com/assets/10526591/24231049/a8c06dac-0fc5-11e7-9003-e61d92abd1b6.png "Warped"
[image6]: https://cloud.githubusercontent.com/assets/10526591/24231641/258e5d96-0fc9-11e7-8209-59232aa9c9c7.png "Window"
[image7]: https://cloud.githubusercontent.com/assets/10526591/24231642/258f5746-0fc9-11e7-863f-b6fb9929c56a.png "Area"
[image8]: https://cloud.githubusercontent.com/assets/10526591/24231846/583f3a2a-0fca-11e7-984c-6fea7baa58a7.png "Unwarped"
[video]: https://youtu.be/E1UGFOa2ado "Video"

### Deliverables

#### 1. File Explanations

My project includes the following files (in image processing order):
* `calibrate.py` to calibrate the camera using provided chessboard images.
* `undistort.py` to undistort test images using the result from `calibrate.py`.
* `threshold.py` to apply various threshold methods and return thresholded binary output image.
* `perspective.py` to transfrom perspective of the image to bird's eye view.
* `polyfit.py` to detect lane lines, calculate curvature and calculate vehicle position from center of the lane.
* `polydraw.py` to draw the detected lane markings and transform the image back to original image space.
* `video.py` to produce the video with lane markings, curvature value and vehicle position value.
* `project_video_output.mp4` is the final ouput video

#### 2. How to produce output video with lane markings
```sh
python video.py
```

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `calibrate.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objp_list` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgp_list` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objp_list` and `imgp_list` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The calibration result is saved as pickle data for later use.

The `undistort()` function in `undistort.py` reads in the calibration results to undistort the images using the `cv2.undistort()` function. 

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.
The image on the left is the original image and on the right is the undistorted image.

![original][image1]
![undistorted][image2]

*The difference is subtle but notice the different location and shape of the white vehicle in the undistorted image.*

#### 2. Threshold Transformations
I used a combination of color and gradient thresholds to generate a binary image (lines #111 ~ #130 in `threshold.py`). 
I explored various combinations of thresholding methods and the final combination was based on the following logic:
```
[(abs_thresh == 1 | ((mag_thresh == 1) & (dir_thresh == 1))) | hsv_thresh == 1]
```
Note that I used the HSV color space to threshold images with pre-defined ranges for the colors yellow and white(lines #90 ~ #108)
Here are the images for the different threshold methods with the final result shown as the last image named "combined."

![Binary][image3]

#### 3. Perspective Transformation

The code for my perspective transform is done with the `warp()` method inside the class `Perspective`, which appears in lines 33 through 40 in the file `perspective.py`. The `warp()` method takes as inputs an image (`img`) to output a bird's eye view image. I chose the source and destination points as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 700      | 260, 700        | 
| 1080, 700      | 1020, 700      |
| 570, 720     | 240, 0      |
| 710, 460      | 1040, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. 

![original][image4]
![warped][image5]

#### 4. Lane line detection
In order to identiy and organize pixels to the left line and right line, I first took a histogram along all the columns in the lower half of the image. With this histogram I added up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I then used a sliding window placed around the line centers, to find and follow the lines up to the top of the frame (`poly_fit_slide()` method in lines #63 ~ #144 in `polyfit.py`).

![alt text][image6]

Since I now know where the lines are I don't need to do a blind search again for the next frame. I searched in a margin around the previous line position (`poly_fit_skip()` method in lines #29 ~ #63 in `polyfit.py`).

![alt text][image7]

#### 5. Radius of Curvature and Vehicle Position from Center

I calculated the radius of curvature with the `curvature()` method in lines #149 ~ #178 in `polyfit.py`. I assumed the lane is about 30 meters long and 3.7 meters wide when converting pixel values to real world space.

I also calculated the vehicle position from center with the `vehicle_position()` method in lines #180 ~ #191 in `polyfit.py`. The method calculates the difference between the center of the lane and center of the image.

#### 6. Lane Markings on Original Image Space

I drew the lane markings and warped back the image to the original image space with the `draw()` function in lines #12 ~ #37 in `polydraw.py`. Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

The output video is `project_video_output.mp4`.

The **[video]** is also available on Youtube


---

### Discussion

#### 1. Limiatations of Current Pipeline

As the pipeline relies soley on camera input, it can struggle in tougher conditions such as in snow or in low-light conditions.
I can try making the pipeline more robust by using more complicated thresholding methods and/or adding recovery options in case lane detection fails in one frame.

