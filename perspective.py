import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from threshold import combined_threshold


def perspective_transform(img):
    """
    Transform perspective based on source points and destination points
    """

    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[200, 700],
         [1080, 700],
         [570, 460],
         [710, 460]]
    )

    dst = np.float32(
        [[260, 700],
         [1020, 700],
         [240, 0],
         [1040, 0]]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size)
    unwarped = cv2.warpPerspective(warped, M_inv, (warped.shape[1], warped.shape[0]))

    return warped, unwarped, M, M_inv


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

    # Transform perspective
    warped1, unwarped1, M1, M_inv1 = perspective_transform(abs_thresh)
    warped2, unwarped2, M2, M_inv2 = perspective_transform(mag_thresh)
    warped3, unwarped3, M3, M_inv3 = perspective_transform(dir_thresh)
    warped4, unwarped4, M4, M_inv4 = perspective_transform(hls_thresh)
    warped5, unwarped5, M5, M_inv5 = perspective_transform(hsv_thresh)
    warped6, unwarped6, M6, M_inv6 = perspective_transform(combined_output)

    # Plot binary output images in order
    plt.subplot(2, 3, 1)
    plt.title("abs")
    plt.imshow(warped1, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title("mag")
    plt.imshow(warped2, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title("dir")
    plt.imshow(warped3, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title("HLS")
    plt.imshow(warped4, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title("HSV")
    plt.imshow(warped5, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title("combined")
    plt.imshow(warped6, cmap='gray')

    plt.show()

    plt.imshow(image)
    plt.show()
