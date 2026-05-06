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
    thresh, binary = cv.threshold(frame[:cutoff], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    y_positions = get_strata_positions(binary, num_strata)
    cv.imshow("binary", binary)

    prev_terrain = None

    for idx, y_pos in enumerate(y_positions):
        if y_pos > cutoff: continue
        if idx != 14 and idx != 13: continue

        layer = binary[y_pos]
        valleys, plateaus, terrain = get_terrain(layer, y_pos)

        if prev_terrain is not None:
            resp = is_similar_terrain(prev_terrain, terrain)
            cv.putText(draw, resp, (0, y_pos), cv.FONT_HERSHEY_PLAIN, 1.0, (0,255,0))

            if resp.startswith("# of chunks"):
                print("prev:", prev_terrain)
                print("curr:", terrain)
                print("\n")
        prev_terrain = terrain

        for (start, end, y_pos, is_valley, *rest) in terrain:
            cv.line(draw, (start, y_pos), (end, y_pos), (255, 0, 0) if is_valley else (255, 0, 255))


def is_similar_terrain(t1, t2):
    # similar terrain should have the same number of chunks
    if len(t1) != len(t2):
        return "# of chunks" + "(" + str(len(t1)) + ", " + str(len(t2)) + ")"
    
    terrain_width = t1[-1][1]
    
    for i in range(len(t1)):
        land1 = t1[i]
        land2 = t2[i]
        (start1, end1, y_pos1, is_valley1, *rest) = land1
        (start2, end2, y_pos2, is_valley2, *rest) = land2

        # similar chunks should have same parity
        if is_valley1 != is_valley2:
            return "parity"
        
        # skip valley chunks
        if is_valley1:
            continue

        # similar chunks should have roughly the same width (within 10% of the average of the 2)
        width1 = end1 - start1
        width2 = end2 - start2
        avg_width = (width1 + width2) / 2
        if abs(avg_width - width1) > 10: #and min(width1, width2) / avg_width < 0.9:
            return "width"

        # similar chunks should start at roughly the same place (within 2% of the frame width) We expect 52 keys at most, so this is not entirely arbitrary
        if abs(start2 - start1) > avg_width / 2:
            return "position"
        
    return "match"


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


    # frame = cv.GaussianBlur(frame, (5,5), 0)
    cv.line(frame, (0, cutoff), (width, cutoff), (0,0,255), 3) # type: ignore
    # frame = frame[:cutoff]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    # cropped = cv.GaussianBlur(cropped, (5,5), 0)

    # binarize_strata(cropped, frame, 20)
    binarize_whole(value, frame, 20)

    cv.imshow("frame", frame)
    return paused

# setup_video_capture(branch, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(branch, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(branch, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(branch, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
setup_video_capture(branch, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")