import math
import asyncio
import cv2 as cv
import numpy as np
from scipy import stats
from setup import setup_video_capture

scan_progress = 0
votes = {}
vote_threshold = 120
vote_verdict = None
num_strata = 25

def process(frame):
    paused = False
    (height, width, channels) = frame.shape
    scale = 720 / height
    frame = cv.resize(frame, None, fx=scale, fy=scale)
    (height, width, channels) = frame.shape
    global num_strata

    if vote_verdict is None:
        scan(frame)
    else:
        num_votes, layers = votes[vote_verdict]
        for strata_num, layer in layers.items():
            step = (height // 2) / (num_strata + 1)
            y_pos = math.floor(step * (strata_num + 1)) + (height // 2)
            draw_terrain(frame, layer, y_pos, (255,0,0))

    cv.imshow("frame", frame)

    return paused


def scan(frame, num_strata = 25, batch = 5):
    global scan_progress
    global votes
    global vote_threshold
    global vote_verdict

    (height, width, channels) = frame.shape
    layers, positions = stratify(frame, num_strata, top=height//2)

    for i in range(batch * scan_progress, batch * (scan_progress + 1)):
        layer = layers[i]
        valleys, plateaus = adaptive_quantization(layer)
        y_pos = positions[i]
        valid = is_valid(plateaus)

        color = (0,255,0) if valid else (0,0,255)
        draw_terrain(frame, plateaus, y_pos, color)

        if valid:
            num_votes = cast_vote(votes, plateaus, i)
            print("strata", str(i), ":", len(plateaus), num_votes)

            if num_votes >= vote_threshold:
                vote_verdict = len(plateaus)
                print("vote decided for", vote_verdict)
                break

    scan_progress = (scan_progress + 1) % (len(layers) // batch)
    

def draw_terrain(frame, terrain, y_pos, color):
    for (start, end) in terrain:
        cv.rectangle(frame, (start, y_pos - 5), (end, y_pos + 5), color, 2)

    cv.rectangle(frame, (0, y_pos), (frame.shape[1], y_pos), (100,0,0), 1) # type: ignore


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


def adaptive_quantization(image):
    (colors, labels) = quantize_colors(image, 3)
    valleys, plateaus = get_terrain(labels)

    # if plateau are not uniform, retry using a different k
    if not is_uniform(plateaus):
        (colors, labels) = quantize_colors(image, 2)
        valleys, plateaus = get_terrain(labels)

    if not is_uniform(plateaus):
        (colors, labels) = quantize_colors(image, 4)
        valleys, plateaus = get_terrain(labels)

    return (valleys, plateaus)


def is_valid(plateaus):
    return is_uniform(plateaus) and len(plateaus) >= 7


def is_uniform(terrain, buffer = 1, scale_thresh = 1.5, pixel_thresh = 8):
    shortest = math.inf
    longest = 0.0

    if len(terrain) <= 2 * buffer + 1:
        return False

    # # we don't know the full extent of the first and last runs, so we exclude them from consideration
    for i in range(buffer, len(terrain) - buffer):
        (start, end) = terrain[i]
        dist = end - start

        if dist > longest:
            longest = dist

        if dist < shortest:
            shortest = dist

    if (longest / shortest) >= scale_thresh and (longest - shortest) > pixel_thresh:
        return False
    
    return True


def get_terrain(labels, min_plat_size = 4, min_valley_size = 2):
    terrain = labels.flatten()
    plateau_label, _ = stats.mode(terrain)
    in_valley = False

    valleys = []
    plateaus = []
    start_of_valley = 0
    start_of_plateau = 0
    
    for i in range(len(labels)):
        if not in_valley and terrain[i] != plateau_label:
            in_valley = True
            start_of_valley = i
            if (i - start_of_plateau) >= min_plat_size:
                plateaus.append((start_of_plateau, i))

        elif in_valley and terrain[i] == plateau_label:
            in_valley = False
            start_of_plateau = i
            if (i - start_of_valley) >= min_valley_size:
                valleys.append((start_of_valley, i))

    if in_valley and (len(terrain) - start_of_valley) >= min_valley_size:
        valleys.append((start_of_valley, len(terrain)))
    elif not in_valley and (len(terrain) - start_of_plateau) >= min_plat_size:
        plateaus.append((start_of_plateau, len(terrain)))

    return valleys, plateaus


# first indexed by plateau count, then by strata number
def cast_vote(votes, plateaus, strata_num):
    choice = len(plateaus)

    # when we count a new number of keys, file it under history
    if choice not in votes.keys():
        votes[choice] = [1, {strata_num: plateaus}]

    # when we count a number of keys we've seen before, update history
    else:
        num_votes, strata_data = votes[choice]
        votes[choice][0] = num_votes + 1
        strata_data[strata_num] = plateaus

    return (votes[choice][0])


# setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(process, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")