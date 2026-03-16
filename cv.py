import cv2 as cv
import numpy as np

frame = None

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


bg_sub = cv.createBackgroundSubtractorKNN()
bg_sub.setDist2Threshold(3600)
bg_sub.setHistory(1)

def process(frame: MatLike): # type: ignore
    frame = cv.resize(frame, None, fx=0.34, fy=0.34)
    cv.imshow("frame", frame)
    no_bg = bg_sub.apply(frame)
    cv.imshow("no bg", no_bg)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    cv.imshow("binary", binary)
    return


setup_video_capture(process=process, path_to_video="videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")