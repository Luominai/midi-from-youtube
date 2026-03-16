import cv2 as cv
import numpy as np

from key_segmentation import find_black_keys, find_black_keys_grayscale, blur_size, upper_thresh, lower_thresh

history = {}
current_truth = None
voting_finished = False
vote_threshold = 90


def setup_video_capture(process, path_to_video=""):
    is_paused = True
    skip_frame = True
    frame_count = 0
    global frame

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
        
        process(frame)

        skip_frame = False


def process(frame: MatLike): # type: ignore
    global current_truth
    global voting_finished
    global vote_threshold

    frame = cv.resize(frame, None, fx=0.8, fy=0.8)

    if not voting_finished:
        blur = cv.GaussianBlur(frame, (blur_size, blur_size), 0)
        black_keys = find_black_keys(blur)
        num_keys_detected = len(black_keys)

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
                print("switch to ", num_keys_detected)

            # lock when one set reaches the vote threshold
            if (old_count + 1) >= vote_threshold:
                print("voting finished")
                voting_finished = True

        # print(num_keys_detected, history[num_keys_detected][1])


    for key in current_truth[0]: # type: ignore
        (x, y, width, height, area) = key
        color = (0, 255, 0) if voting_finished else (0, 0, 255)
        cv.circle(frame, (x, y), 5, color)
        cv.rectangle(frame, (x, y), (x + width, y + height), color, 1)


    cv.imshow("frame", frame)
    return


setup_video_capture(process=process, path_to_video="videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")