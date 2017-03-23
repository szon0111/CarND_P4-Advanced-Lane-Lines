import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undistort import undistort
from threshold import combined_threshold
from perspective import Perspective
from polyfit import Polyfit


def draw(img, warped, left_fit, right_fit, Minv):
    """
   Draw the driveable area and warp back to original image space
   Combine the result with undistorted image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Generate x and y values for plotting
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result


if __name__ == '__main__':
    polyfit = Polyfit()
    perspective = Perspective()

    image_file = 'test_images/test2.jpg'
    image = mpimg.imread(image_file)
    undistorted = undistort(image)
    # Create binary outputs
    abs_thresh, mag_thresh, dir_thresh, hls_thresh, hsv_thresh, combined_output = combined_threshold(undistorted)
    plt.imshow(combined_output, cmap='gray')
    warped = perspective.warp(combined_output)
    plt.imshow(warped, cmap='gray')
    # Find lanes
    left_fit, right_fit, vars = polyfit.poly_fit_skip(warped)
    new_img = draw(image, warped, left_fit, right_fit, perspective.Minv)
    plt.imshow(new_img)
    plt.show()
    print(new_img.shape)
