import copy
import math

import cv2 as cv
import numpy as np

from key_segmentation import find_black_keys, find_black_keys_grayscale, blur_size, sort_into_buckets, upper_thresh, lower_thresh, get_pattern, get_leftmost_note
from utils import label_keys, split_line, to_gray

history = {}
current_truth = None
voting_finished = False
voting_enabled = False
vote_threshold = 30
area_std_thresh = 85
black_key_upper_thresh = 30
steps_allowed = 4

line_thresh = 30


def setup_video_capture(process, path_to_video=""):
    is_paused = True
    skip_frame = True
    frame_count = 0
    global frame
    global line_thresh

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
        elif key == ord('z'):
            line_thresh -= 1
            print("line thresh: ", line_thresh)
        elif key == ord('x'):
            line_thresh += 1
            print("line thresh: ", line_thresh)
        if is_paused and not skip_frame:
            continue

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if process(frame):
            is_paused = True

        skip_frame = False


def process(frame: MatLike): # type: ignore
    global current_truth
    global voting_finished
    global vote_threshold
    global black_key_upper_thresh
    global voting_enabled
    pause = False

    frame = cv.resize(frame, None, fx=0.6, fy=0.6)

    if not voting_finished:
        blur = cv.GaussianBlur(frame, (blur_size, blur_size), 0)
        black_keys = find_black_keys(blur, upper_thresh=black_key_upper_thresh)
        num_keys_detected = len(black_keys)

        # if the variance in area is high, our threshold probably doesn't work well for this video
        area_std = np.std(black_keys[:,4])
        if area_std > area_std_thresh:
            voting_enabled = False
            step = 1
            for i in range(steps_allowed):
                black_key_upper_thresh += step
                new_keys = find_black_keys(blur, upper_thresh=black_key_upper_thresh)
                new_area_std = np.std(new_keys[:,4])
                print("thresh:", black_key_upper_thresh, "std:", new_area_std)

                if new_area_std >= area_std:
                    step *= -1

                if new_area_std < area_std_thresh:
                    break

        if area_std <= area_std_thresh:
            voting_enabled = True


        color = (255, 0, 0)
        draw_keys(black_keys, frame, color)


        # voting may be disabled due to adjustment
        if voting_enabled:
            # when we count a new number of keys, file it under history
            if num_keys_detected not in history.keys():
                history[num_keys_detected] = [black_keys, 1.0]
                # establish ground truth if there is none yet
                if current_truth == None:
                    current_truth = history[num_keys_detected]
            # when we count a number of keys we've seen before, update history
            else:
                historical_avg = history[num_keys_detected][0]                                  # get the historical avg
                old_count = history[num_keys_detected][1]                                       # get the number of frames in the history
                history[num_keys_detected][1] += 1                                              # increment the number of frames in the history
                historical_avg = (historical_avg * old_count + black_keys) / (old_count + 1)    # update history

                # switch ground truths if this number has been voted more
                if old_count + 1 > current_truth[1]: # type: ignore
                    current_truth = history[num_keys_detected]
                    print("switch to", num_keys_detected)

                # lock when one set reaches the vote threshold
                if (old_count + 1) >= vote_threshold:
                    print("voting finished")
                    voting_finished = True

            # print(num_keys_detected, history[num_keys_detected][1])

    if current_truth:
        color = (0, 255, 0) if voting_finished else (0, 0, 255)
        keys = current_truth[0]
        draw_keys(current_truth[0], frame, color)

        # chunk_gray = cv.cvtColor(chunk, cv.COLOR_BGR2GRAY)
        # ret,chunk_binary = cv.threshold(chunk_gray, 100, 255, cv.THRESH_BINARY)
        # cv.rectangle(frame, (0, avg_y), (frame.shape[1], avg_y + 50))

        if voting_finished:
            avg_y = math.ceil(np.average(keys[:,1] + keys[:,3]))
            avg_height = math.ceil(np.average(keys[:,3]))

            # print(avg_y, avg_height)

            # avg_y: int = math.floor(np.average(current_truth[0][:,1]))
            # avg_height: int = math.floor(np.average(current_truth[0][:,3]))
            offset = 5
            scale = 0.3

            chunk = frame[avg_y + offset : avg_y + math.ceil(avg_height * scale) + offset]

            # cv.imshow("chunk", chunk)

            upper_line = frame[avg_y + offset]

            # cv.imshow("upper", upper_line)
            # lower_line = frame[avg_y + math.ceil(avg_height * scale) + offset]

            upper_split = split_line(upper_line, spike_thresh=15, plat_thresh=5)
            print(upper_split.shape)

            plotted = []

            for (start, end) in upper_split:
                print(start, end, end - start)
                # cv.circle(frame, (start + (end - start) // 2, avg_y + offset), 5, (255,255,0), 1)

                color = [255,255,0]
                y = avg_y + offset + 8

                for (p_s, p_e) in plotted:
                    if abs(start - p_s) < 5 and abs(end - p_e) < 5:
                        color[2] += 30
                        y += 4

                        # print(color)
                cv.circle(frame, (start, y), 5, copy.deepcopy(color), 1)
                plotted.append((start, end))

            # print(upper_split)

            # cv.rectangle(
            #     frame, 
            #     (0, avg_y + offset), 
            #     (frame.shape[1], avg_y + math.ceil(avg_height * scale) + offset), 
            #     (255,0,255), 
            #     1
            # )

            pause = True


    #     if voting_finished:
    #         order = np.argsort(current_truth[0][:,0])
    #         sorted_keys = current_truth[0][order] # type: ignore
    #         mid = len(sorted_keys) // 2
    #         sample = sorted_keys[mid-2: mid+3]

    #         # print(get_leftmost_note(sorted_keys))
    #         # draw_keys(sample, frame, (255,0,255))
    #         label_keys(sorted_keys, frame)

    #         pause = True

    cv.imshow("frame", frame)
    return pause


def draw_keys(keys, frame, color):
    for key in keys: # type: ignore
        (x, y, width, height, area) = key
        cv.circle(frame, (x, y), 5, color)
        cv.rectangle(frame, (x, y), (x + width, y + height), color, 1)


# setup_video_capture(process=process, path_to_video="videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
setup_video_capture(process=process, path_to_video="videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")