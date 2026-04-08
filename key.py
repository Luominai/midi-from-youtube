from math import floor

import numpy as np
import keyboard_parser2 as KeyboardParser
import cv2 as cv

octave_colors = [(180,0,0),(0,180,0),(0,0,180),(180,180,0),(0,180,180),(180,0,180)]

class Key:
    def __init__(self, frame, strata, note, octave, on_press, on_release):
        self.strata = strata[np.argsort(strata["y_pos"])]
        avg_width = np.average([(start - end) for (start, end, *rest) in self.strata])
        points = np.array([((start + end) // 2, y_pos) for (start, end, y_pos, *rest) in self.strata])
        (vx, vy, x0, y0) = cv.fitLine(points, distType=cv.DIST_L2, param=0, reps=0.01, aeps=0.01) # type: ignore
        scale = 200
        x_adj = round((avg_width * vx)[0])
        self.line_start = (floor((x0 - vx * scale)[0]) + x_adj, floor((y0 - vy * scale)[0]))
        self.line_end = (floor((x0 + vx * scale)[0]) + x_adj, floor((y0 + vy * scale)[0]))
        self.note = note
        self.octave = octave
        self.on_press = on_press
        self.on_release = on_release
        self.is_pressed = False
        self.base_color = get_average_color(frame, strata)
        self.history = np.full(shape=(len(strata),3,3), fill_value=self.base_color)
        self.vote_threshold = 10
        self.current_votes = 0
        self.vote_finished = False
        self.verdict = None
        self.distance_threshold = 120


    def process(self, frame):
        num_activations = 0
        KeyboardParser.draw_terrain(frame, self.strata, octave_colors[self.octave % len(octave_colors)], size=2, draw_text=False)
        cv.line(frame, self.line_start, self.line_end, (255,0,0), 1) # type: ignore

        # print("========" + self.note + str(self.octave) + "========")
        for idx, (start, end, y_pos, *rest) in enumerate(self.strata):
            pixels = frame[y_pos][start : end]
            strata_average = np.average(pixels, axis=0)

            if self.vote_finished and self.verdict is not None:
                color_distance = get_color_distance(strata_average, self.verdict[idx])
                # print(color_distance)
                if color_distance > self.distance_threshold:
                    num_activations += 1
                continue

            if not self.vote_finished:
                distances = []
                self.history[idx] = np.roll(self.history[idx], 1, axis=0)
                self.history[idx][0] = np.average(pixels, axis=0)
                strata_historical_average = np.average(self.history[idx], axis=0)

                distance = get_color_distance(strata_average, strata_historical_average)
                distances.append(distance)
                # print(idx, distance)

                # TODO: Each strata needs to decide independently
                if np.average(distances) < 1:
                    self.current_votes += 1
                    # print("votes:", self.current_votes)
                else:
                    self.current_votes = 0
                    # print('reset')

                if self.current_votes > (self.vote_threshold * len(self.strata)):
                    self.vote_finished = True
                    self.verdict = self.history
                    # print("decided:", self.history)

        if self.verdict is None:
            return

        # print(num_activations, len(self.strata))
        if not self.is_pressed and num_activations >= 0.6 * len(self.strata):
            self.is_pressed = True
            self.on_press(self)

        if self.is_pressed and num_activations < 0.6 * len(self.strata):
            self.is_pressed = False
            self.on_release(self)
        

def get_average_color(frame, strata):
    num_pixels = 0

    avg = np.zeros(shape=3)

    for (start, end, y_pos, *rest) in strata:
        pixels = frame[y_pos][start: end]
        for pixel in pixels:
            avg += pixel
            num_pixels += 1
            # print("values:", b,g,r)
            # print("current sum:", sum_b, sum_g, sum_r)

    return avg / num_pixels


def get_color_distance(c1, c2):
    return np.linalg.norm(c1 - c2)


def get_unit_vector(v1):
    length = np.linalg.norm(v1)
    return v1 / length


def get_angle_between(v1, v2):
    v1_unit = get_unit_vector(v1)
    v2_unit = get_unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))


def get_orientation(strata, scale = 1.0):
    """
    Returns an (x, y) pair representing the x and y diff from the top to the bottom strata
    """
    (top_start, top_end, top_y_pos, *top_rest) = strata[0]
    (bottom_start, bottom_end, bottom_y_pos, *bottom_rest) = strata[-1]
    top_center = top_start + (top_end - top_start) // 2
    bottom_center = bottom_start + (bottom_end - bottom_start) // 2

    return (
        floor(scale * (bottom_center - top_center)), 
        floor(scale * (bottom_y_pos - top_y_pos))
    )


def as_point(stratum):
    return (floor(stratum[0] + (stratum[1] - stratum[0]) // 2), stratum[2])