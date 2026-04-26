from math import ceil

import cv2 as cv
from keyboard_parser4 import KeyboardParser4, resize
from setup import setup_video_capture
import numpy as np

parser = KeyboardParser4()
cutoff = None

def branch(frame):
    if len(parser.keys) == 0:
        parser.process(frame)
    else:
        process(frame)

def local_otsu(frame, rows, cols):
    (height, width, *rest) = frame.shape
    chunk_width = width / cols
    chunk_height = height / rows

    # break the image into strips
    strips = []
    for row_i in range(rows):
        start_h = round(row_i * chunk_height)
        end_h = round((row_i + 1) * chunk_height)
        strip = frame[start_h : min(end_h, height)]

        # print("strip:", strip.shape)

        # break the strips into chunks
        chunks = []
        for col_i in range(cols):
            start_w = round(col_i * chunk_width)
            end_w = round((col_i + 1) * chunk_width)
            chunk = strip[:,start_w : min(end_w, width)]

            # perform otsu on the chunk
            _, chunk = cv.threshold(chunk, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            chunks.append(chunk)

        # reform the binarized strip by hstacking chunks
        strip = np.hstack(chunks)
        strips.append(strip)
        

    # reform the binarized image by vstacking strips
    image = np.vstack(strips)
    return image


def double_otsu(frame):
    # Threshold once using Otsu
    thresh, binary = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # use the binary image as a mask on the original (grayscale) image
    masked = cv.bitwise_and(binary, frame)

    # filter to keep only the pixels that didn't survive the first thresholding
    pixels = masked.flatten()
    filter = pixels == 0
    pixels = pixels[filter]

    # then threshold the remaining pixels again to find an Otsu value
    thresh, _ = cv.threshold(pixels,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # use the second Otsu value to threshold the original image and return the result
    ret, binary = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)
    return binary

    
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
    frame = frame[:cutoff]
    cropped = frame
    
    hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    cropped = value

    # cropped = cv.medianBlur(cropped, 5)
    # cropped = cv.GaussianBlur(cropped, (3,3), 0)
    cropped = cv.GaussianBlur(cropped, (7,7), 0)
    cropped = cv.GaussianBlur(cropped, (7,7), 0)
    thresh, cropped = cv.threshold(cropped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,7))
    # cropped = cv.erode(cropped, kernel)
    (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(cropped, connectivity=4)
    for (left, top, width, height, area) in stats:
        cv.rectangle(frame, (left, top), (left + width, top + height), (0,255,0))


    cv.imshow("frame", frame)
    cv.imshow("cropped", cropped)
    # cv.imshow("hue", hue)
    # cv.imshow("saturation", saturation)
    cv.imshow("value", value)
    return paused

# setup_video_capture(branch, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(branch, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
setup_video_capture(branch, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(branch, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture(branch, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")