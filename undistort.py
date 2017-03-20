import glob
import cv2
import pickle
import numpy as np


# Read in saved object points and image points
dist_pickle = pickle.load(open('./calibrate.p', "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# Make a list of test images
image_list = glob.glob('./test_images/test*.jpg')
for i, filename in enumerate(image_list):
    # read in image
    img = cv2.imread(filename)
    # undistort image
    print('working on', filename)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # save undistored image
    write_name = './output_images/undistorted' + str(i + 1) + '.jpg'
    cv2.imwrite(write_name, img)
