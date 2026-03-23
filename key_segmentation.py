import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

image_path = "frames/key_hit.jpg"

# numbers that seem to work
# lower    upper    scale   blur_size   method              use case
# 104      172      .34     3           canny then blur     edge detection
# 132      179      .34     3           canny then blur     edge detection
# 136      328      .8      3           canny then blur     edge detection
# 203      181      .8      5           blur then canny     edge detection
# 203      181      .8      5           blur then mask      black key identification

lower_thresh = 203
upper_thresh = 181
scale = .8
blur_size = 11

def main():
    image = cv.imread(image_path)
    global lower_thresh
    global upper_thresh
    global scale 
    global blur

    if image is None:
        print('Error opening image!')
        return
    
    image = cv.resize(image, None, fx=scale, fy=scale)
    print(image.shape)

    while True:
        blur = cv.GaussianBlur(image, (blur_size, blur_size), 0)
        # canny = cv.Canny(blur, lower_thresh, upper_thresh, None, 3)

        black_keys = find_black_keys(blur)
        get_pattern(black_keys)

        for key in black_keys:
            (x, y, width, height, area) = key
            cv.circle(image, (x, y), 5, (255, 0, 0))

        cv.imshow("Source", image)
        # cv.imshow("Blur post Canny", canny)

        key = cv.waitKey()
        if key == ord('q'):
            break
        elif key == ord('a'):
            lower_thresh -= 1
        elif key == ord('d'):
            lower_thresh += 1
        elif key == ord('s'):
            upper_thresh -= 1
        elif key == ord('w'):
            upper_thresh += 1

        print("lower: ", lower_thresh, " upper: ", upper_thresh)


def edge_detection(image: MatLike, lower, upper, blur_size, scale = 1.0, method = 0): # type: ignore
    output = cv.resize(image, None, fx=scale, fy=scale)

    if method == 0:
        output = cv.GaussianBlur(image, (blur_size, blur_size), 0)
        output = cv.Canny(output, lower, upper, None, 3)
    elif method == 1:
        output = cv.Canny(image, lower, upper, None, 3)
        output = cv.GaussianBlur(output, (blur_size, blur_size), 0)

    return output


def get_keys(image: NDArray): # type: ignore
    pass


def filter_color(image: NDArray, lower, upper):
    mask = cv.inRange(image, lower, upper)
    return cv.bitwise_and(image, image, mask=mask)


# data must be sorted
# spike threshold indicating the largest diff between adjacent values we'll tolerate in a bucket
# range threshold that say show big a range we'll tolerate in a bucket
# [lower bound, upper bound, start index (inclusive), end index (exclusive), length]
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
# [x, y, width, height, area]
def find_black_keys(image, lower_thresh = 0, upper_thresh = 30):
    black_mask = cv.inRange(image, np.array([lower_thresh, lower_thresh, lower_thresh]), np.array([upper_thresh, upper_thresh, upper_thresh]))
    black_mask_components = cv.connectedComponentsWithStats(black_mask)
    
    (num_labels, labels, stats, centroids) = black_mask_components

    order = np.argsort(stats[:,1]) # sort components by x-value
    buckets = sort_into_buckets(stats[order][:,1], spike_thresh=20, range_thresh=30) # get buckets
    bucket_with_most_values = buckets[np.argmax(buckets, axis=0)[4]] # find the bucket with most items
    (lb, ub, start, end, length) = bucket_with_most_values

    return stats[start : end] # returns items in the bucket containing black keys


def find_black_keys_grayscale(image, lower_thresh = 0, upper_thresh = 30):
    black_mask = cv.inRange(image, lower_thresh, upper_thresh) # type: ignore
    black_mask_components = cv.connectedComponentsWithStats(black_mask)
    (num_labels, labels, stats, centroids) = black_mask_components

    order = np.argsort(stats[:,1]) # sort components by x-value
    buckets = sort_into_buckets(stats[order][:,1]) # get buckets
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


# main()