import cv2 as cv
import numpy as np
from scipy import stats


def setup_video_capture(process, path_to_video):
    is_paused = True
    skip_frame = True
    frame_count = 0
    frame = None

    cap = cv.VideoCapture(path_to_video)
    while cap.isOpened():
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('s'):
            is_paused = not is_paused
        elif key == ord('d'):
            skip_frame = True
            
        if key == ord('f') and frame is not None: # save frame
            output_filename = "frames/" + str(frame_count) + ".jpg"
            cv.imwrite(output_filename, frame)
            print("saved frame to ", output_filename)
            frame_count += 1

        if is_paused and not skip_frame:
            continue

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        is_paused = process(frame)

        skip_frame = False


def process(frame):
    scale = 1152 / frame.shape[1]
    pause = False
    frame = cv.resize(frame, None, fx=scale, fy=scale)

    image = binarize(frame)
    cv.imshow("frame", image)

    return pause


def binarize(frame):
    scale = 1152 / frame.shape[1]
    frame = cv.resize(frame, None, fx=scale, fy=scale)
    frame = frame[50:150]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # threshold once
    thresh, otsu = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # get only the parts that survive the threshold
    frame = cv.bitwise_and(otsu, gray)
    flat = frame.flatten()
    filter = flat > 0
    flat = flat[filter]
    # threshold those parts again
    thresh, _ = cv.threshold(flat,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret, frame = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)


    # thresh, frame = cv.threshold(gray,thresh,255,cv.THRESH_BINARY)
    return frame


# setup_video_capture("videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture("videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture("videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture("videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# image1 = cv.imread('frames/keyboard1.jpg')
# cv.imshow("keyboard1", binarize(image1, type=0))
# cv.imshow("keyboard2", binarize(image1, type=1))
# cv.imshow("keyboard3", binarize(image1, type=2))
# image2 = cv.imread('frames/keyboard2.jpg')
# cv.imshow("keyboard2", binarize(image2))
image3 = cv.imread('frames/keyboard3.jpg')
cv.imshow("keyboard1", binarize(image3))
cv.imshow("keyboard2", binarize(image3))
cv.imshow("keyboard3", binarize(image3))
cv.waitKey(-1)