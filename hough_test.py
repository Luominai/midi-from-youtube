from math import ceil
import math

import cv2 as cv
from keyboard_parser4 import KeyboardParser4, get_strata_positions, get_terrain, resize, stratify, stratify_gray
from setup import setup_video_capture
import numpy as np

parser = KeyboardParser4()
cutoff = None
hough = None
max_val = 0

def branch(frame):
    if len(parser.keys) == 0:
        parser.process(frame)
    else:
        process(frame)


def binarize_strata(frame, draw, num_strata):
    height, width, *rest = frame.shape
    y_positions = get_strata_positions(frame, num_strata)

    for y_pos in y_positions:
        layer = frame[y_pos]

        thresh, binary = cv.threshold(layer, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        if thresh < 30:
            cv.line(draw, (0, y_pos), (width, y_pos), (255, 0, 0))

        valleys, plateaus, terrain = get_terrain(binary, y_pos)

        for (start, end, y_pos, is_valley, *rest) in terrain:
            cv.line(draw, (start, y_pos), (end, y_pos), (255, 0, 0) if is_valley else (0, 0, 255))


def binarize_whole(frame, draw, num_strata):
    global hough
    global max_val
    global cutoff

    height, width, *rest = frame.shape
    thresh, binary = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    y_positions = get_strata_positions(binary, num_strata)
    # cv.imshow("binary", binary)

    for y_pos in y_positions:
        if y_pos > cutoff: continue

        layer = binary[y_pos]
        valleys, plateaus, terrain = get_terrain(layer, y_pos)

        for (start, end, y_pos, is_valley, *rest) in terrain:
            # cv.line(draw, (start, y_pos), (end, y_pos), (255, 0, 0) if is_valley else (255, 0, 255))

            if hough[y_pos][start] < 255: # type: ignore
                hough[y_pos][start] += 1 # type: ignore
            # max_val = hough[y_pos][start] if hough[y_pos][start] > max_val else max_val # type: ignore

            if hough[y_pos][end - 1] < 255: # type: ignore
                hough[y_pos][end - 1] += 1 # type: ignore
            # max_val = hough[y_pos][end - 1] if hough[y_pos][end - 1] > max_val else max_val # type: ignore


def draw_hough_lines(frame):
    global hough

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # thresh, _ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # space = cv.Canny(frame, thresh / 2, thresh)[:cutoff]
    # cv.imshow("canny", space)
    space = hough[:300]
    # space = hough[:cutoff] * (255 / max(255,max_val)) # type: ignore
    # space = np.astype(space, np.uint8)

    lines = cv.HoughLines(
        space,  # type: ignore
        rho=20, 
        theta=(math.pi / 180), 
        threshold=100,
        min_theta=0, 
        max_theta=(math.pi / 90)
    ) # type: ignore

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(frame, pt1, pt2, (0,255,0), 3, cv.LINE_AA)

    cv.imshow("hough", hough[:300])

    # lines = cv.HoughLines(
    #     space,  # type: ignore
    #     rho=10, 
    #     theta=(math.pi / 360), 
    #     threshold=10,
    #     min_theta=(89 * math.pi / 90), 
    #     max_theta=(math.pi)
    # ) # type: ignore

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(frame, pt1, pt2, (0,255,0), 3, cv.LINE_AA)


    
def process(frame):
    global cutoff
    global hough
    paused = False
    frame = resize(frame)
    (height, width, channels) = frame.shape

    if hough is None:
        hough = np.zeros(shape=(height, width, 1), dtype=np.uint8)

    if cutoff is None:
        avg_y = 0
        avg_black = 0
        num_black = 0
        for key in parser.keys:
            key.scale = 100
            key.process(frame)
            
            avg_y += int(key.strata[0]["y_pos"])
            if key.strata[0]["note"][-1] == "#":
                avg_black -= int(key.strata[0]["start"])
                avg_black += int(key.strata[0]["end"])
                num_black += 1
        
        avg_y = avg_y // len(parser.keys)
        cutoff = avg_y - 15
        print(avg_black / num_black)


    cv.line(frame, (0, cutoff), (width, cutoff), (0,0,255), 3) # type: ignore
    # frame = frame[:cutoff]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    # cropped = cv.GaussianBlur(cropped, (5,5), 0)

    # binarize_strata(cropped, frame, 20)
    binarize_whole(value, frame, 20)
    draw_hough_lines(frame)

    cv.imshow("frame", frame)
    # cv.imshow("cropped", cropped)
    # cv.imshow("value", value)
    # cv.imshow("hough", hough)
    return paused

# setup_video_capture(branch, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(branch, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
setup_video_capture(branch, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(branch, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture(branch, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")