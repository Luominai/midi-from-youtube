from math import ceil

import cv2 as cv
from keyboard_parser4 import KeyboardParser4, get_strata_positions, get_terrain, resize, stratify, stratify_gray
from setup import setup_video_capture
import numpy as np

parser = KeyboardParser4()
cutoff = None

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
    height, width, *rest = frame.shape
    thresh, binary = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    y_positions = get_strata_positions(binary, num_strata)
    cv.imshow("binary", binary)

    for y_pos in y_positions:
        layer = binary[y_pos]
        valleys, plateaus, terrain = get_terrain(layer, y_pos)

        for (start, end, y_pos, is_valley, *rest) in terrain:
            cv.line(draw, (start, y_pos), (end, y_pos), (255, 0, 0) if is_valley else (255, 0, 255))

    
def process(frame):
    global cutoff
    paused = False
    frame = resize(frame)
    (height, width, channels) = frame.shape

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

    cv.imshow("frame", frame)
    # cv.imshow("cropped", cropped)
    cv.imshow("value", value)
    return paused

# setup_video_capture(branch, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(branch, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
setup_video_capture(branch, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(branch, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture(branch, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")