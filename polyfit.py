import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undistort import undistort
from threshold import combined_threshold
from perspective import Perspective


class Polyfit:
    """
    Find and fit poly lines in warped image
    Calculate curvature
    Calculate vehicle distance from center
    """

    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_x = None
        self.right_x = None
        self.left_y = None
        self.right_y = None
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.margin = 80
    def poly_fit_skip(self, img):
        """
        Skip the sliding windows if polynomial is empty and search in a margin around the previous line
        Fit second order polynomial to each line
        """
        # Check if fit line exists
        if self.left_fit is None or self.right_fit is None:
            return self.poly_fit_slide(img)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = self.margin
        left_lane_inds = ((nonzero_x > (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] - margin)) & (
            nonzero_x < (self.left_fit[0] * (nonzero_y**2) + self.left_fit[1] * nonzero_y + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzero_x > (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] - margin)) & (
            nonzero_x < (self.right_fit[0] * (nonzero_y**2) + self.right_fit[1] * nonzero_y + self.right_fit[2] + margin)))
        # Extract left and right line pixel positions
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
        self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        # Variables to use for visualization
        vars = {}
        vars['left_fit'] = self.left_fit
        vars['right_fit'] = self.right_fit
        vars['nonzero_x'] = nonzero_x
        vars['nonzero_y'] = nonzero_y
        vars['left_lane_inds'] = left_lane_inds
        vars['right_lane_inds'] = right_lane_inds

        return self.left_fit, self.right_fit, vars

    def poly_fit_slide(self, img):
        """
        Take a histogram along all the columns in the lower half of the image
        Implement sliding window to find and follow lines up to the top of the image
        Fit second order polynomial to each line
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        num_windows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] / num_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Current positions to be updated for each window
        left_x_current = left_x_base
        right_x_current = right_x_base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = left_x_current - margin
            win_xleft_high = left_x_current + margin
            win_xright_low = right_x_current - margin
            win_xright_high = right_x_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
        self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        # Variables to use for visualization
        vars = {}
        vars['left_fit'] = self.left_fit
        vars['right_fit'] = self.right_fit
        vars['nonzero_x'] = nonzero_x
        vars['nonzero_y'] = nonzero_y
        vars['left_lane_inds'] = left_lane_inds
        vars['right_lane_inds'] = right_lane_inds
        vars['out_img'] = out_img
        vars['margin'] = margin

        return self.left_fit, self.right_fit, vars

    def curvature(self):
        """
        Compute radius of curvature of the fit
        """
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix of the
        # line base position in each case (x=200 for left, and x=900 for right)
        left_x = np.array([200 + (y**2) * quadratic_coeff + np.random.randint(-50, high=51)
                           for y in ploty])
        right_x = np.array([900 + (y**2) * quadratic_coeff + np.random.randint(-50, high=51)
                            for y in ploty])
        left_x = left_x[::-1]  # Reverse to match top-to-bottom in y
        right_x = right_x[::-1]  # Reverse to match top-to-bottom in y
        # Define y-value where we want radius of curvature (max y,
        # corresponding to the bottom of the image)
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(
            ploty * self.ym_per_pix, left_x * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            ploty * self.ym_per_pix, right_x * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix +
                                left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix +
                                 right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        return left_curve_rad, right_curve_rad

    def vehicle_position(self, img):
        """
        Calculate vehicle distance from center
        """
        pos_leftx = self.left_fit[0] * (img.shape[0] - 1) ** 2 + \
            self.left_fit[1] * (img.shape[0] - 1) + self.left_fit[2]
        pos_rightx = self.right_fit[0] * (img.shape[0] - 1) ** 2 + \
            self.right_fit[1] * (img.shape[0] - 1) + self.right_fit[2]
        position = ((img.shape[1] / 2) -
                    ((pos_leftx + pos_rightx) / 2)) * self.xm_per_pix

        return position

    def visualize_window(self, img, vars, save_image=False):
        """
        Visualize the sliding windows and mark the lane lines on bird's eye view image
        """
        # Define variables from vars
        self.left_fit = vars['left_fit']
        self.right_fit = vars['right_fit']
        nonzero_x = vars['nonzero_x']
        nonzero_y = vars['nonzero_y']
        left_lane_inds = vars['left_lane_inds']
        right_lane_inds = vars['right_lane_inds']
        out_img = vars['out_img']
        # Generate x and y values for plotting
        plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit_x = self.left_fit[0] * plot_y**2 + \
            self.left_fit[1] * plot_y + self.left_fit[2]
        right_fit_x = self.right_fit[0] * plot_y**2 + \
            self.right_fit[1] * plot_y + self.right_fit[2]
        # Color in left and right line pixels
        out_img[nonzero_y[left_lane_inds],
                nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds],
                nonzero_x[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        # save visualization
        if save_image:
            plt.savefig('./output_images/vis_window.png', bbox_inches='tight')
        plt.show()

    def visualize_area(self, img, vars, save_image=False):
        """
        Visualize the targeted search area based on previous frame
        """
        # Define variables from vars
        self.left_fit = vars['left_fit']
        self.right_fit = vars['right_fit']
        nonzero_x = vars['nonzero_x']
        nonzero_y = vars['nonzero_y']
        left_lane_inds = vars['left_lane_inds']
        right_lane_inds = vars['right_lane_inds']
        out_img = vars['out_img']
        margin = vars['margin']
        # Generate x and y values for plotting
        plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit_x = self.left_fit[0] * plot_y**2 + \
            self.left_fit[1] * plot_y + self.left_fit[2]
        right_fit_x = self.right_fit[0] * plot_y**2 + \
            self.right_fit[1] * plot_y + self.right_fit[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzero_y[left_lane_inds],
                nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds],
                nonzero_x[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        # save visualization
        if save_image:
            plt.savefig('./output_images/vis_area.png', bbox_inches='tight')
        plt.show()


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
    left_curve_rad, right_curve_rad = polyfit.curvature()
    position = polyfit.vehicle_position(warped)
    print('curvature: {:.2f}m'.format((left_curve_rad + right_curve_rad) / 2))
    print('vehicle position: {:.2f}m from center'.format(position))
    # Visualize lane finding
    polyfit.visualize_window(warped, vars, save_image=True)
    polyfit.visualize_area(warped, vars, save_image=True)
