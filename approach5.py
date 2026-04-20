import cv2 as cv
import numpy as np

from setup import setup_video_capture
import keyboard_parser2 as parser

MOG2_subtractor = cv.createBackgroundSubtractorMOG2(detectShadows = True, history=400, varThreshold=25) # exclude shadow areas from the objects you detected
KNN_subtractor = cv.createBackgroundSubtractorKNN(dist2Threshold=400, detectShadows=False)
bg_sub = MOG2_subtractor

def process(image):
    (height, width, *rest) = image.shape # type: ignore
    scale = 1280 / width
    image = cv.resize(image, None, fx=scale, fy=scale) # type: ignore

    # orig = image[250:470]
    image = image[250:470]
    color = image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image = image[250:470]
    # image = cv.blur(image, (5,5))
    image = cv.GaussianBlur(image, (15, 15), 0)
    # image = cv.medianBlur(image, 5)
    # image = cv.bilateralFilter(image, 9, 75, 75)
    # image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    threshold, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    image = cv.Canny(image, 0, threshold, None, 3)
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(color, contours, -1, (0,255,0), 3)


    # times = 1
    # size = 5
    # for i in range(times):
    #     image = cv.blur(image, (size, size))

    # times = 3
    # size = 3
    # for i in range(times):
    #     image = cv.blur(image, (size, size))

    # (colors, best_labels) = parser.quantize_colors(image, 2)
    # image = colors[best_labels].reshape((image.shape[0], image.shape[1], 3))

    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ############
    # image = bg_sub.apply(image)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # image = cv.dilate(image, kernel=kernel, iterations=10)

    # image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
    # image = cv.blur(image, (5,5))
    # image = image[:,:,1]
        
    # times = 3
    # size = 3
    # for i in range(times):
    #     image = cv.blur(image, (size, size))

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # image = cv.erode(image, kernel=kernel)
    ###############
    # threshold, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # image = cv.Canny(image, 0, threshold, None, 3)
    cv.imshow("a", color)
    cv.imshow("b", image)
    # cv.imshow("b", orig)

# process(cv.imread('frames/birdbrain_crop.jpg'))
# cv.waitKey(-1)
setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(process, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")
# setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")