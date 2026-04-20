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
    image = image[250:470]
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    image = cv.GaussianBlur(image, (5,5), 0)
    layers, positions = parser.stratify(image, 50)
    color = image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    for i in range(len(layers)):
        layer = layers[i]
        y_pos = positions[i]
        
        threshold, layer = cv.threshold(image[y_pos], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        layer = layer.flatten()
        
        # [ <start>, <end>, <is_valley>, <note>, <octave> ]
        terrain = []
        start = 0
        prev = layer[0]
        for i, num in enumerate(layer):
            if num != prev:
                terrain.append((start, i))
                start = i
            prev = num
        terrain.append((start, len(layer)))

        for i, (start, end) in enumerate(terrain):
            if i % 2 == 0:
                cv.rectangle(color, (start, y_pos), (end, y_pos), (0,255,0), 2)
            else:
                cv.rectangle(color, (start, y_pos), (end, y_pos), (255,0,0), 2)
        # start = -1
        # for 


        # (colors, labels) = parser.quantize_colors(layer, 2)
        # valleys, plateaus, full_survey = parser.get_terrain(labels, y_pos, min_plat_size=0, min_valley_size=0)
        # valid = parser.is_valid(plateaus, valleys, full_survey)

        # parser.draw_terrain(image, plateaus, color = (0,128,0))
        # parser.draw_terrain(image, valleys, color = (128,0,0))

    # image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
    # # image = cv.GaussianBlur(image, (5,5), 0)
    # image = cv.bilateralFilter(image, 9, 75, 75)
    # hue = image[:,:,0]
    # saturation = image[:,:,1]
    # value = image[:,:,2]

    # threshold, otsu = cv.threshold(value, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("value", value)
    # cv.imshow("otsu", otsu)
    cv.imshow("image", color)


# setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(process, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")
# setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")