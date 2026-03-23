import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from cv import draw_keys
from key_segmentation import find_black_keys, get_leftmost_note

# image_path = "frames/perspective.jpg"
image_path = "frames/black_bg.jpg"

sharps = ["A", "C", "D", "F", "G"]

def main():
    scale = .67
    blur_size = 11
    image = cv.imread(image_path)
    if image is None:
        print('Error opening image!')
        return
    
    image = cv.resize(image, None, fx=scale, fy=scale)
    blur = cv.GaussianBlur(image, (blur_size, blur_size), 0)

    black_keys = find_black_keys(blur)
    # avg_y = math.ceil(np.average(black_keys[:,1] + black_keys[:,3]))
    # avg_height = math.ceil(np.average(black_keys[:,3]))

    order = np.argsort(black_keys[:,0])
    black_keys = black_keys[order] # type: ignore
    leftmost_note = get_leftmost_note(black_keys)
    print(leftmost_note)
    index = sharps.index(leftmost_note)

    white_keys = []

    for key in black_keys:
        note = sharps[index]
        (x, y, width, height, area) = key
        print(index, note)
        cv.putText(image, note, (x, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        white_keys.append(
            (x - width - 3, y + 2* height // 3, width, height // 3, 2 * area / 9)
        )
        if note == "A" or note == "D":
            white_keys.append(
                (x + width + 3, y + 2* height // 3, width, height // 3, 2 * area / 9)
            )
            if note == "A":
                cv.putText(image, "B", (x + width + 3, 20 + y + 2 * height // 3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif note == "D":
                cv.putText(image, "E", (x + width + 3, 20 + y + 2* height // 3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv.putText(image, sharps[index - 1], (x - width - 3, 20 + y + 2* height // 3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        index = (index + 1) % len(sharps)

    draw_keys(black_keys, image, (0,0,255))
    draw_keys(white_keys, image, (255,0,0))

    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(blur, 50, 0, None, 3)
    # ret,thresh = cv.threshold(gray,200,255,cv.THRESH_BINARY)
    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(image, contours, -1, (255,0,0), 2)
    
    # cv.watershed(image, contours)

    # chunk = image[avg_y + 5 : avg_y + 5 + avg_height // 2]
    # chunk_gray = cv.cvtColor(chunk, cv.COLOR_BGR2GRAY)
    # ret,chunk_binary = cv.threshold(chunk_gray, 100, 255, cv.THRESH_BINARY)
    # chunk_edges = cv.Canny(chunk_binary, 10, 0)
    # chunk_contours, heirarchy = cv.findContours(chunk_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(chunk, chunk_contours, -1, (0,255,0), 1)

    # cv.imshow("chunk", chunk)
    # cv.imshow("chunk edges", chunk_edges)
    # chunk_edges = edges[avg_y : avg_y + avg_height // 3]


    # cv.rectangle(image, (0, avg_y + 5), (image.shape[1], avg_y + avg_height // 3), (255,0,0))

    cv.imshow("image", image)
    # cv.imshow("thresh", thresh)
    # cv.imshow("canny", edges)
    cv.waitKey(0)
    


main()