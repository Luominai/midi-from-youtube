import math
import asyncio
import cv2 as cv
import numpy as np
from scipy import stats
from setup import setup_video_capture

def process(frame):
    paused = False
    (height, width, channels) = frame.shape
    scale = 720 / height
    frame = cv.resize(frame, None, fx=scale, fy=scale)

    (height, width, channels) = frame.shape
    layers, positions = stratify(frame, 3, top=(2 * height // 3), offset=(0 * height // 32))

    for i in range(len(layers)):
        layer = layers[i]
        (colors, labels) = quantize_colors(layer, 2)
        valleys = get_valleys(labels)
        y_pos = positions[i]

        for (start, end) in valleys:
            print(start, end)
            cv.rectangle(frame, (start, y_pos - 5), (end, y_pos + 5), (0,0,255), 2)

        print("strata " + str(i) + ":", len(valleys))

    for pos in positions:
        cv.rectangle(frame, (0, pos), (frame.shape[1], pos), (100,0,0), 1) # type: ignore

    cv.imshow("frame", frame)


    return paused


def stratify(frame, num_layers, top = 0, offset = 0):
    (height, width, channels) = frame.shape
    layers = np.empty(shape=(num_layers, 1, frame.shape[1], 3))
    positions = np.empty(shape=num_layers, dtype=int)
    step = (height - top) / (num_layers + 1)

    for i in range(num_layers):
        y_pos = math.floor(step * (i + 1)) + top + offset
        positions[i] = y_pos
        layers[i] = frame[y_pos]

    return layers, positions


def quantize_colors(image, num_colors):
    # take the image and reshape it into a 1d array of BGR tuples
    data = np.float32(image).reshape((image.shape[0] * image.shape[1], 3))

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    retval, best_labels, centers = cv.kmeans(
        data=data,  # type: ignore
        K=num_colors,
        bestLabels=None, # type: ignore
        criteria=criteria, 
        attempts=5, 
        flags=cv.KMEANS_RANDOM_CENTERS
    )

    colors = np.uint8(centers)

    return (colors, best_labels)


def get_valleys(labels):
    terrain = labels.flatten()
    plateau, _ = stats.mode(terrain)
    in_valley = False

    valleys = []
    start_of_valley = 0
    
    for i in range(len(labels)):
        if not in_valley and terrain[i] != plateau:
            in_valley = True
            start_of_valley = i

        if in_valley and terrain[i] == plateau:
            in_valley = False
            valleys.append((start_of_valley, i))

    if in_valley:
        valleys.append((start_of_valley, len(terrain)))

    return valleys


setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(process, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")