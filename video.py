import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

from undistort import undistort
from threshold import combined_threshold
from perspective import Perspective
from polyfit import Polyfit
from polydraw import draw


polyfit = Polyfit()
perspective = Perspective()


def video(input_video):
    """

    """
    original_video = '{}.mp4'.format(input_video)
    video = VideoFileClip(original_video)
    output_video = video.fl_image(process)
    output_video.write_videofile('{}_output.mp4'.format(input_video), audio=False)


def process(video):
    """

    """
    # Undistort frame
    undistorted = undistort(video)
    # Threshold frame
    abs_thresh, mag_thresh, dir_thresh, hls_thresh, hsv_thresh, combined_output = combined_threshold(undistorted)
    # Change perspective
    warped = perspective.warp(combined_output)
    # Fit polynomial lines
    left_fit, right_fit, vars = polyfit.poly_fit_skip(warped)
    frame = draw(undistorted, warped, left_fit, right_fit, perspective.Minv)
    # Annotate curvature to video
    left_curve_rad, right_curve_rad = polyfit.curvature()
    cv2.putText(frame, 'curvature: {:.2f}m'.format((left_curve_rad + right_curve_rad) / 2),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)
    # Annotate distance from center to video
    position = polyfit.vehicle_position(warped)
    cv2.putText(frame, 'vehicle position: {:.2f}m from center'.format(position),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)

    return frame


if __name__ == '__main__':
    video('project_video')
