import glob
import pickle
import cv2

def undistort(img):
    # Read in saved object points and image points
    dist_pickle = pickle.load(open('./calibrate.p', "rb"))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    img = cv2.undistort(img, mtx, dist, None, mtx)

    return img


if __name__ == '__main__':
    # Make a list of test images
    image_list = glob.glob('./test_images/*.jpg')
    for i, filename in enumerate(image_list):
        # read in image
        img = cv2.imread(filename)
        # undistort image
        print('working on', filename)
        undistorted = undistort(img)
        # save undistored image
        write_name = './output_images/undistorted' + str(i + 1) + '.jpg'
        cv2.imwrite(write_name, undistorted)
