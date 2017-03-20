import cv2
import glob
import pickle
import numpy as np


# Prepare object points - (0, 0, 0), (1, 0, 0) .... (8, 5, 0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and images points from all images
objp_list = []  # 3D points in real world space
imgp_list = []  # 2D points in image plane

# Make a list of calibration images
image_list = glob.glob('./camera_cal/calibration*.jpg')

# Go through the list and search for chessboard corners
for i, filename in enumerate(image_list):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        print('working on', filename)
        objp_list.append(objp)
        imgp_list.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = './output_images/corners_found' + str(i + 1) + '.jpg'
        cv2.imwrite(write_name, img)

# Load image for reference
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Perform calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objp_list, imgp_list, img_size, None, None)

# Save the  calibration result for later use
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('./calibrate.p', 'wb'))
