import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_threshold(img, orient='x', thresh=(20, 100)):
    """
    Apply Sobel x or y, take absolute value and apply threshold.
    Output array of the same size as the input image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient and take absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a binary mask where scaled gradient magnitude thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Apply Sobel x or y, compute magnitude value and apply threshold.
    Output array of the same size as the input image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary mask where overall magnitude thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Apply Sobel x or y, compute direction of gradient and apply threshold.
    Output array of the same size as the input image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the
    # gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def hls_threshold(img, thresh=(100, 255)):
    """
    Separate S channel from HLS color space and apply threshold
    Output array of the same size as the input image
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Separate S channel
    s_channel = hls[:, :, 2]
    # Create a binary mask where color thresholds are met
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output


def hsv_threshold(img, thresh=([20, 100, 100], [35, 255, 255], [0, 0, 230], [180, 25, 255])):
    """
    Find white and yellow based on range and apply threshold
    Output array of the same size as the input image
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2. COLOR_RGB2HSV)
    # define hsv ranges for yellow and white
    lower_yellow = np.array(thresh[0])
    upper_yellow = np.array(thresh[1])
    lower_white = np.array(thresh[2])
    upper_white = np.array(thresh[3])
    # threshold hsv_img with defined ranges
    yellow_hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_hsv = cv2.inRange(hsv, lower_white, upper_white)
    binary_output = np.zeros_like(hsv[:, :, 0])
    binary_output[((yellow_hsv != 0) | (white_hsv != 0))] = 1

    return binary_output


def combined_threshold(img):
    """
    Apply combined thresholds using pre-defined functions
    Output array of the same size as the input image
    """
    abs_thresh = abs_threshold(img, orient='x', thresh=(20, 100))
    mag_thresh = mag_threshold(img, sobel_kernel=3, mag_thresh=(50, 100))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.2))
    hls_thresh = hls_threshold(img, thresh=(200, 255))
    hsv_thresh = hsv_threshold(img, thresh=(
        [20, 100, 100], [35, 255, 255], [0, 0, 230], [180, 25, 255]))

    # Create the final binary mask where combined thresholds are met
    combined_output = np.zeros_like(hls_thresh)
    combined_output[(abs_thresh == 1 | ((mag_thresh == 1) &
                                        (dir_thresh == 1))) | hsv_thresh == 1] = 1
    # combined_output[(abs_thresh == 1 | ((mag_thresh == 1) &
    #                                     (dir_thresh == 1))) | hls_thresh == 1] = 1

    return abs_thresh, mag_thresh, dir_thresh, hls_thresh, hsv_thresh, combined_output


if __name__ == '__main__':
    with open('./calibrate.p', "rb") as pickle_file:
        dist_pickle = pickle.load(pickle_file)
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    image_file = 'test_images/test2.jpg'
    image = mpimg.imread(image_file)
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # Create binary outputs
    abs_thresh, mag_thresh, dir_thresh, hls_thresh, hsv_thresh, combined_output = combined_threshold(image)

    # Plot binary output images in order
    plt.subplot(2, 3, 1)
    plt.title("abs")
    plt.imshow(abs_thresh, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title("mag")
    plt.imshow(mag_thresh, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title("dir")
    plt.imshow(dir_thresh, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title("HLS")
    plt.imshow(hls_thresh, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title("HSV")
    plt.imshow(hsv_thresh, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title("combined")
    plt.imshow(combined_output, cmap='gray')

    plt.show()
