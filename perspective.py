import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from undistort import undistort
from threshold import combined_threshold


class Perspective:
    """
    Transform perspective based on source points and destination points
    """

    def __init__(self):
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

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        """
        warp image to bird's eye view
        """
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size)

        return warped

    def unwarp(self, img):
        """
       unwarp image back to original view
        """
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.M, img_size)

        return unwarped


if __name__ == '__main__':
    perspective = Perspective()


    image_file = 'test_images/test2.jpg'
    image = mpimg.imread(image_file)
    undistorted = undistort(image)
    # Create binary outputs
    abs_thresh, mag_thresh, dir_thresh, hls_thresh, hsv_thresh, combined_output = combined_threshold(undistorted)
    # Transform perspective
    warped1 = perspective.warp(abs_thresh)
    warped2 = perspective.warp(mag_thresh)
    warped3 = perspective.warp(dir_thresh)
    warped4 = perspective.warp(hls_thresh)
    warped5 = perspective.warp(hsv_thresh)
    warped6 = perspective.warp(combined_output)
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
