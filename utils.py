import math

import numpy as np
import cv2 as cv

sharps = ["A", "C", "D", "F", "G"]

def sort_into_buckets(data, spike_thresh = 8, range_thresh = 10):
    start_of_bucket = 0
    buckets = []

    for i in range(len(data)):
        current = data[i]
        previous = data[i - 1] if i > 0 else None
        lowerb = data[start_of_bucket]

        # if exceed spike threshold or range threshold, create new bucket
        if (previous and current - previous > spike_thresh) or (current - lowerb > range_thresh):
            buckets.append(
                (data[start_of_bucket], data[i], start_of_bucket, i, i - start_of_bucket)
            )
            start_of_bucket = i
            
    # add the last threshold 
    buckets.append(
        (data[start_of_bucket], data[-1], start_of_bucket, len(data), len(data) - start_of_bucket)
    )

    return np.array(buckets)


# Given an image, returns stats of the components identified as black keys
# This method uses a mask to find black colored components and will fail when color changes
# should modify this to fall back on history when the function fails to find something in the current frame
# [x, y, width, height, area, note]
def find_black_keys(image, lower_thresh = 0, upper_thresh = 30):
    black_mask = cv.inRange(image, np.array([lower_thresh, lower_thresh, lower_thresh]), np.array([upper_thresh, upper_thresh, upper_thresh]))
    black_mask_components = cv.connectedComponentsWithStats(black_mask)
    
    (num_labels, labels, stats, centroids) = black_mask_components

    order = np.argsort(stats[:,1]) # sort components by x-value
    buckets = sort_into_buckets(stats[order][:,1], spike_thresh=20, range_thresh=30) # get buckets
    bucket_with_most_values = buckets[np.argmax(buckets, axis=0)[4]] # find the bucket with most items
    (lb, ub, start, end, length) = bucket_with_most_values

    return stats[start : end] # returns items in the bucket containing black keys


def get_pattern(keys, offset = 0):
    mid = len(keys) // 2 + offset
    sample = keys[mid-2 : mid+3]
    pattern = "1"

    gaps = []
    for i in range(4):
        curr = sample[i]
        next = sample[i + 1]
        gaps.append(next[0] - curr[0])

    for i in range(1, len(gaps)):
        curr = gaps[i]
        prev = gaps[i - 1]

        # special case to account for the second key
        if i == 1:
            if (prev == max(curr, prev) and prev > 1.1 * curr):
                pattern += " "
            pattern += "1"  

        if curr == max(curr, prev) and curr > 1.1 * prev:
            pattern += " "

        pattern += "1"    

    print(pattern)

    return pattern


# keys must be sorted
def get_leftmost_note(keys):
    pattern = get_pattern(keys)
    sharps = ""

    match pattern:
        case "111 11":
            sharps = "FGACD"
        case "11 11 1":
            sharps = "GACDF"
        case "1 11 11":
            sharps = "ACDFG"
        case "11 111":
            sharps = "CDFGA"
        case "1 111 1":
            sharps = "DFGAC"

    # the first key of the sample is at index i
    i = len(keys) // 2 - 2
    # the first key of the keyboard is at index 0
    # iterate backward i times
    # or calculate i % 5 and go back that amount
    first_note = sharps[-1 * (i % 5)]
    return first_note


def draw_keys(keys, frame, color):
    for key in keys: # type: ignore
        (x, y, width, height, area) = key
        cv.circle(frame, (x, y), 5, color)
        cv.rectangle(frame, (x, y), (x + width, y + height), color, 1)


def label_keys(black_keys, image):
    leftmost_note = get_leftmost_note(black_keys)
    index = sharps.index(leftmost_note)
    white_keys = []

    extra_offset = 0
    scale = 0.8

    for key in black_keys:
        note = sharps[index]
        (x, y, width, height, area) = key
        print(index, note)
        cv.putText(image, note, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        white_keys.append(
            (x - width - extra_offset, y, math.floor(width * scale), height, area)
        )
        if note == "A" or note == "D":
            white_keys.append(
                (x + width + extra_offset, y, math.floor(width * scale), height, area)
            )
            if note == "A":
                cv.putText(image, "B", (x + width + extra_offset, 20 + y + height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif note == "D":
                cv.putText(image, "E", (x + width + extra_offset, 20 + y + height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv.putText(image, sharps[index - 1], (x - width - extra_offset, 20 + y + height), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        index = (index + 1) % len(sharps)

    draw_keys(white_keys, image, (255, 0, 0))

    

def get_white_keys(black_keys):
    order = np.argsort(black_keys[:,0])
    black_keys = black_keys[order] # type: ignore
    leftmost_note = get_leftmost_note(black_keys)
    index = sharps.index(leftmost_note)

    white_keys = []

    for key in black_keys:
        note = sharps[index]
        (x, y, width, height, area) = key

        white_keys.append(
            (x - width - 3, y + 2* height // 3, width, height // 3, 2 * area / 9)
        )
        if note == "A" or note == "D":
            white_keys.append(
                (x + width + 3, y + 2* height // 3, width, height // 3, 2 * area / 9)
            )

        index = (index + 1) % len(sharps)

    return white_keys