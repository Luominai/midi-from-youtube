import math
import asyncio
import cv2 as cv
import numpy as np
from scipy import stats
from setup import setup_video_capture

scan_progress = 0

def process(frame):
    paused = False
    (height, width, channels) = frame.shape
    scale = 720 / height
    frame = cv.resize(frame, None, fx=scale, fy=scale)
    global scan_progress

    (height, width, channels) = frame.shape
    # blur = cv.blur(frame, (3,3))
    
    layers, positions = stratify(frame, 25, top=height//2)
    batch = 5

    for i in range(batch * scan_progress, batch * (scan_progress + 1), 1):
        layer = layers[i]
        (colors, labels) = quantize_colors(layer, 4)
        valleys = get_valleys(labels)
        y_pos = positions[i]

        for (start, end) in valleys:
            cv.rectangle(frame, (start, y_pos - 5), (end, y_pos + 5), (0,0,255), 2)

        cv.rectangle(frame, (0, y_pos), (frame.shape[1], y_pos), (100,0,0), 1) # type: ignore

        print("strata", str(i), ":", len(valleys))

    scan_progress = (scan_progress + 1) % (len(layers) // batch)

    cv.imshow("frame", frame)


    return paused


def stratify(frame, max_layers, top = 0, offset = 0, limit = None, reverse = False):
    limit = max_layers if limit is None else limit
    (height, width, channels) = frame.shape
    layers = np.empty(shape=(limit, 1, frame.shape[1], 3))
    positions = np.empty(shape=limit, dtype=int)
    step = (height - top) / (max_layers + 1)

    for i in range(limit):
        step_num = i + max_layers - limit + 1 if reverse else i + 1
        y_pos = math.floor(step * step_num) + top + offset
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


# for syncing the output of get_valleys across multiple strata
# each strata will have a different number of valleys so the input will be jagged
# numpy does not support jagged arrays, so the input will be a python list
def sync_strata_valleys(valleys):
    # the value at index i is the current index of valley[i]
    index_of = np.zeros(shape=len(valleys), dtype=int)

    # get the elements
    elements = [valley[index_of[valley_id]] for valley_id, valley in enumerate(valleys)]

    # once we get the elements, we can group them according to their start value using kmeans (or a simple threshold)

    # choose the group that has the smallest mean start so we don't skip over any values

    # group the elements again by width

    # record the data in each subgroup as part of one whole

    # advance the indexes of the arrays that the data in the group came from

    return np.array(elements)

        
# valleys = [
#     np.array([0, 2, 4, 6]),
#     np.array([1, 3, 5]),
#     np.array([6, 7]),
# ]

# print(sync_strata_valleys(valleys))

# setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(process, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
# setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")