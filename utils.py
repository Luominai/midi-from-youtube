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
def find_black_keys(image, lower_thresh = 0, upper_thresh = 30, blur_size = 11):
    blur = cv.GaussianBlur(image, (blur_size, blur_size), 0)
    black_mask = cv.inRange(blur, np.array([lower_thresh, lower_thresh, lower_thresh]), np.array([upper_thresh, upper_thresh, upper_thresh]))
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


def split_line(line, spike_thresh=30, plat_thresh = 5):
    start_of_bucket = 0
    buckets = []

    print(line.shape)

    cliff = []
    in_valley = False

    while (np.average(line[start_of_bucket] < 180)):
        start_of_bucket += 1

    for i in range(start_of_bucket, len(line)):
        current = line[i].astype(np.int16)

        # if there is no cliff, establish the current point as the cliff
        if len(cliff) == 0 and not in_valley:
            cliff = current

        # compare the current point to the cliff
        distance = np.linalg.norm(current - cliff)

        # if the elevation difference is high, we are entering a valley
        if not in_valley and distance > spike_thresh:
            in_valley = True
            # store the current bucket
            buckets.append((start_of_bucket, i))

        # while in a valley, skip buckets and instead check for when we exit the valley
        if in_valley:
            # we exit the valley if our new ground is within some elevation of the old ground
            if distance < plat_thresh:
                in_valley = False
                start_of_bucket = i
            else:
                # print("valley:", current, cliff, current - cliff if i > 0 else None, distance)
                continue

        # while not in a valley, update the cliff so its elevation is averaged
        cliff = (cliff * (i - start_of_bucket) + current) / (i - start_of_bucket + 1)
        # print("plateau:", current, cliff, current - cliff if i > 0 else None, distance)
            
    # add the last bucket if we're not in a valley
    if not in_valley:
        buckets.append(
            (start_of_bucket, len(line))
        )

    return np.array(buckets)


def to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def to_binary(image, threshold = 127):
    return cv.threshold(image, threshold, 255, cv.THRESH_BINARY)





def process_video(path_to_video):
    # vars for handling video control
    is_paused = True
    skip_frame = True
    frame_count = 0
    frame = None
    
    # setup a video stream
    cap = cv.VideoCapture(path_to_video)
    while cap.isOpened():
        # get keyboard input
        key = cv.waitKey(1)

        # handle exit, pause, and skip inputs
        if key == ord('q'):
            break
        elif key == ord('s'):
            is_paused = not is_paused
        elif key == ord('d'):
            skip_frame = True

        # handle save frame input
        if key == ord('f') and frame is not None: # save frame
            output_filename = "frames/" + str(frame_count) + ".jpg"
            cv.imwrite(output_filename, frame)
            print("saved frame to ", output_filename)
            frame_count += 1

        # if we are paused and we're not skipping the current frame, do not read the next frame yet
        if is_paused and not skip_frame:
            continue

        # read and process the frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # pause if the frame returns True
        is_paused = process_frame(frame)

        # if we skipped the most recent frame, reset the skip_frame variable so we don't skip the next frame 
        skip_frame = False
        


def process_frame(frame, known_keys, votes):
    # resize the frame
    frame = cv.resize(frame, None, fx=0.6, fy=0.6)

    # if we don't know where the black keys are yet, vote until we figure it out
    if known_keys == None:
        # we cast a vote for the number of keys detected. If we detect one number consistently, then the video has probably stabilized
        detected_black_keys = find_black_keys(frame)
        vote_bucket = len(detected_black_keys)

        # before we cast the vote, try to validate it. Check for indications that the detected keys might be bad
        (valid, metrics) = validate_black_keys(detected_black_keys)
        
        # if the vote is not valid, make adjustments to improve validity
        if not valid:
            valid = adjust_thresholds(frame, thresholds, metrics)
        # if the vote is valid (after adjustment), cast it
        if valid:
            (num_votes, vote_avg) = cast_vote(votes, vote_bucket, detected_black_keys)

            # if the number of votes for this bucket reaches some threshold, the vote is decided
            if num_votes >= thresholds["required_votes"]:
                known_black_keys = vote_avg
                white_keys = find_white_keys(known_black_keys)

                known_keys = vote_avg


    
    pass


# (x, y, width, height)
def find_black_keys(image, lower_thresh = 0, upper_thresh = 30, blur_size = 11):
    # blur the image
    blur = cv.GaussianBlur(image, (blur_size, blur_size), 0)

    # use a mask to identify black pixels and merge them into components
    black_mask = cv.inRange(blur, np.array([lower_thresh, lower_thresh, lower_thresh]), np.array([upper_thresh, upper_thresh, upper_thresh]))
    black_mask_components = cv.connectedComponentsWithStats(black_mask)
    (num_labels, labels, stats, centroids) = black_mask_components

    # sort components by x-value
    order = np.argsort(stats[:,1]) 

    # group components together according to y-value
    buckets = sort_into_buckets(stats[order][:,1], spike_thresh=20, range_thresh=30) 

    # the bucket with the most items is likely to be one with the keys
    bucket_with_most_values = buckets[np.argmax(buckets, axis=0)[4]]
    (start, end) = bucket_with_most_values

    return stats[start : end][:4]


# data must be sorted
def sort_into_buckets(data, spike_thresh = 8, range_thresh = 10):
    start_of_bucket = 0
    buckets = []
    lower_bound = data[0]

    # skip the first element. The first element will always belong to the first bucket
    for i in range(1, len(data)):
        current = data[i]
        previous = data[i - 1]

        # if exceed spike threshold or range threshold, create new bucket
        if (current - previous > spike_thresh) or (current - lower_bound > range_thresh):
            # save current bucket
            buckets.append(
                (start_of_bucket, i)
            )

            # set the start and lower_bound of the new bucket
            start_of_bucket = i
            lower_bound = data[i]
            
    # add the last threshold 
    buckets.append(
        (start_of_bucket, len(data))
    )

    return np.array(buckets)


def validate_black_keys(black_keys, area_std_thresh = 85):
    verdict = True
    metrics = {}
    
    # keys should have similar area
    metrics["area_std"] = np.std(black_keys[:,2] * black_keys[:,3])
    if metrics["area_std"] > area_std_thresh:
        verdict = False

    # TODO: keys should have a certain width to height ratio

    return (verdict, metrics) 


def adjust_thresholds(image, thresholds, initial_metrics, steps_allowed = 4):
    step_size = 1
    prev_area_std = initial_metrics["area_std"]


    for i in range(steps_allowed):
        # try modifying the thresholds and seeing how the metrics change
        thresholds["black_key_upper"] += step_size
        detected_black_keys = find_black_keys(image, upper_thresh=thresholds["black_key_upper"])
        (valid, metrics) = validate_black_keys(detected_black_keys)

        # if the adjustment makes the detection valid again, break
        if valid:
            return True

        # if the adjustment increases error, change step direction
        if metrics["area_std"] > prev_area_std:
            prev_area_std = metrics["area_std"]
            step_size *= -1

        
    return False


def find_white_keys(image, black_keys, offset = 5, scale = 0.3):
    white_keys = []

    avg_y = math.ceil(np.average(black_keys[:,1] + black_keys[:,3]))
    avg_height = math.ceil(np.average(black_keys[:,3]))

    upper_line = image[avg_y + offset]
    lower_line = image[avg_y + math.ceil(avg_height * scale) + offset]

    upper_split = split_line(upper_line, spike_thresh=20, plat_thresh=5)
    lower_split = split_line(lower_line, spike_thresh=20, plat_thresh=5)

    for i in range(len(upper_split)):
        (upper_start, upper_end) = upper_split[i]
        (lower_start, lower_end) = lower_split[i]

        x = min(upper_start, lower_start)
        y = avg_y + offset
        width = max(upper_end, lower_end) - x
        height = math.ceil(avg_height * scale)

        white_keys.append(
            (x, y, width, height)
        )

    return np.array(white_keys)


# (x, y, width, height, note, octave)
def label_keys(black_keys, white_keys):
    output = []

    # sort black keys by their center x-value
    black_incr_center_x_order = np.argsort(black_keys[:,0] + black_keys[:,2] // 2, 0)
    black_keys = black_keys[black_incr_center_x_order]
    first_black_key = black_keys[0]
    index_of_first_black_key_in_merged_array = -1
    
    # sort white keys by their center-x value
    white_incr_center_x_order = np.argsort(white_keys[:,0] + white_keys[:,2] // 2, 0)
    white_keys = white_keys[white_incr_center_x_order]

    # merge the two according to their center-x values
    black_index = 0
    white_index = 0

    while black_index < len(black_keys) and white_index < len(white_keys):
        black = black_keys[black_index]
        white = white_keys[white_index]

        if black[0] + black[2] // 2 < white[0] + white[2] // 2:
            if black == first_black_key:
                index_of_first_black_key_in_merged_array = len(output)
            output.append(black)
            black_index += 1
        else:
            output.append(white)
            white_index += 1

    for i in range(black_index, len(black_keys)):
        output.append(black_keys[i])

    for i in range(white_index, len(white_keys)):
        output.append(white_keys[i])


    leftmost_black_note = get_leftmost_black_note(black_keys)
    leftmost_note = shift_note(leftmost_black_note, -1 * index_of_first_black_key_in_merged_array)

    notes = "A A# B C C# D D# E F F# G G#".split(" ")
    index_of_leftmost_note = notes.index(leftmost_note)
    curr_index = index_of_leftmost_note
    octave = 0

    dtype = [
        ('x', 'i4'),
        ('y', 'i4'),
        ('width', 'i4'),
        ('height', "i4"),
        ('note', 'U2'),
        ('octave', 'i1')
    ]
    labeled_keys = np.empty(shape=(len(output), 6), dtype=dtype) 
    
    for i in range(len(output)):
        (x, y, width, height) = output[i]
        labeled_keys[i] = (x, y, width, height, notes[curr_index], octave)

        curr_index = (curr_index + 1) % len(notes)
        if notes[curr_index] == "C":
            octave += 1

    return labeled_keys


# keys must be sorted
def get_leftmost_black_note(keys):
    pattern = get_pattern(keys)
    sharps = ""

    match pattern:
        case "111 11":
            sharps = "F# G# A# C# D#"
        case "11 11 1":
            sharps = "G# A# C# D# F#"
        case "1 11 11":
            sharps = "A# C# D# F# G#"
        case "11 111":
            sharps = "C# D# F# G# A#"
        case "1 111 1":
            sharps = "D# F# G# A# C#"

    sharps = sharps.split(" ")

    # the first key of the sample is at index i
    i = len(keys) // 2 - 2
    # the first key of the keyboard is at index 0. calculate i % 5 and go back that amount
    first_black_note = sharps[-1 * (i % 5)]
    return first_black_note


def shift_note(note, amount):
    notes = "A A# B C C# D D# E F F# G G#".split(" ")
    index = notes.index(note)
    index = (index + amount) % len(notes)
    return notes[index]