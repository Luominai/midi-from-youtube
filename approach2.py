import math
import asyncio
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


def color_quantization(frame):
    # take the image and reshape it into a 1d array of BGR tuples
    data = np.float32(frame).reshape((frame.shape[0] * frame.shape[1], 3))

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    retval, best_labels, centers = cv.kmeans(
        data=data,  # type: ignore
        K=4,
        bestLabels=None, # type: ignore
        criteria=criteria, 
        attempts=10, 
        flags=cv.KMEANS_RANDOM_CENTERS
    )

    colors = np.uint8(centers)

    return (None, colors, best_labels)


def binarize(frame, type=0):
    scale = 1152 / frame.shape[1]
    frame = cv.resize(frame, None, fx=scale, fy=scale)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (3,3))

    if type == 0:
        _, image = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return image
    if type == 1:
        image = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2)
        return image
    if type == 2:
        image = custom_binarize(frame)
        return image
    
    return frame


def custom_binarize(frame):
    (_, colors, labels) = color_quantization(frame)

    # print(colors)

    # identify the most common color in the image
    most_common_label, _ = stats.mode(labels, axis=None, keepdims=False) # type: ignore
    most_common_color = colors[most_common_label] # type: ignore

    # print(most_common_color)

    # sort the colors by value
    averages = np.average(colors, 1)
    order = np.argsort(averages)    # gives the indexes of colors in ascending order. first element is the darkest
    sorted_colors = colors[order] # type: ignore

    # print(sorted_colors)

    # find the position of the most common color in the sorted colors array
    pos = find_row(sorted_colors, most_common_color)

    # convert all colors lighter than the most common color into the common color
    for i in range(pos + 1, len(sorted_colors)):
        row = sorted_colors[i]
        row_pos = find_row(colors, row)     # position in unsorted array
        colors[row_pos] = most_common_color # type: ignore

    # print(colors)

    # convert all colors darker than the most common into the darkest color
    for i in range(0, pos):
        row = sorted_colors[i]
        row_pos = find_row(colors, row)     # position in unsorted array
        colors[row_pos] = sorted_colors[0] # type: ignore

    # print(colors)

    # recolor the image
    thresholded_pixels = colors[labels.flatten()] # type: ignore
    thresholded_image = thresholded_pixels.reshape(frame.shape)

    return thresholded_image


def find_row(array, row):
    return np.where(np.all(array == row, axis = 1))[0][0]


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
cv.imshow("keyboard1", binarize(image3, type=0))
cv.imshow("keyboard2", binarize(image3, type=1))
cv.imshow("keyboard3", binarize(image3, type=2))
cv.waitKey(-1)