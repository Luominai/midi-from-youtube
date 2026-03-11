import math

import cv2 as cv
import numpy as np

image_path = "frames/0.jpg"

# numbers that seem to work
# upper    lower
# 104      172
# 132      179

lower_thresh = 130
upper_thresh = 180

def main():
    image = cv.imread(image_path)
    global lower_thresh
    global upper_thresh

    if image is None:
        print('Error opening image!')
        return
    
    image = cv.resize(image, None, fx=0.34, fy=0.34)

    while True:
        blur = cv.GaussianBlur(image, (3,3), 0)
        canny_no_blur = cv.Canny(image, lower_thresh, upper_thresh, None, 3)
        canny_with_blur = cv.Canny(blur, lower_thresh, upper_thresh, None, 3)
        blur_post_canny = cv.GaussianBlur(canny_no_blur, (3,3), 0)
        cv.imshow("Source", image)
        cv.imshow("Canny no blur", canny_no_blur)
        cv.imshow("Canny with blur", canny_with_blur)
        cv.imshow("Blur post Canny", blur_post_canny)
        key = cv.waitKey()

        if key == ord('q'):
            break
        elif key == ord('a'):
            lower_thresh -= 1
        elif key == ord('d'):
            lower_thresh += 1
        elif key == ord('s'):
            upper_thresh -= 1
        elif key == ord('w'):
            upper_thresh += 1

        print("lower: ", lower_thresh, " upper: ", upper_thresh)

def edge_detection(image: MatLike):
    canny = cv.Canny(image, lower_thresh, upper_thresh, None, 3)
    blur = cv.GaussianBlur(canny, (5,5), 0)
    return blur

main()