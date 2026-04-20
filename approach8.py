import cv2 as cv
from keyboard_parser4 import KeyboardParser4, resize
from setup import setup_video_capture

parser = KeyboardParser4()

def branch(frame):
    if len(parser.keys) == 0:
        parser.process(frame)
    else:
        process(frame)
    
def process(frame):
    paused = False
    frame = resize(frame)
    (height, width, channels) = frame.shape

    avg_y = 0
    for key in parser.keys:
        key.scale = 100
        key.process(frame)
        
        avg_y += int(key.strata[0]["y_pos"])
        
    avg_y = avg_y // len(parser.keys)
    cutoff = avg_y - 15
    cv.line(frame, (0, cutoff), (width, cutoff), (0,0,255), 3) # type: ignore
    cropped = frame[:cutoff]

    


    cv.imshow("frame", frame)
    return paused

# setup_video_capture(branch, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")
# setup_video_capture(branch, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
# setup_video_capture(branch, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
# setup_video_capture(branch, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
setup_video_capture(branch, "videos/Flavor Foley - ＂Ego Renegade Boy＂ ｜ (Piano Cover + Sheet Music) [U7pYL8adTNY].webm")