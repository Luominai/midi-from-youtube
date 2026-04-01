import numpy as np


class Key:
    def __init__(self, frame, strata, note, octave, on_press, on_release):
        self.strata = strata
        self.note = note
        self.octave = octave
        self.on_press = on_press
        self.on_release = on_release
        self.is_pressed = False
        self.base_color = get_average_color(frame, strata)
        self.history = np.full(shape=(5,3), fill_value=self.base_color)
        self.distance_threshold = 150



    def process(self, frame):
        current_color = get_average_color(frame, self.strata)
        historical_average = np.average(self.history, axis=0)
        color_distance = get_color_distance(historical_average, current_color)

        if not self.is_pressed:
            if color_distance > self.distance_threshold:
                self.on_press(self)
                self.is_pressed = True
            else:
                self.history = np.roll(self.history, 1, axis=0)
                self.history[0] = current_color

        if self.is_pressed:
            if color_distance < self.distance_threshold:
                self.on_release(self)
                self.is_pressed = False

        # print("average:", historical_average)
        # print("current:", current_color)
        # print(self.note, self.octave)
        # print("dist:", color_distance)
        # print("angle:", color_angle)
        # self.history = np.roll(self.history, 1, axis=0)
        # self.history[0] = current_color
        # print(self.history)

            # print(self.history)
        # else:
            # print(get_color_distance(historical_average, current_color))
            # self.on_press(self)

        # if self.note == "E" and self.octave == 4:
        #     print(angle)

        # color_distance = get_color_distance(current_color, self.base_color)
        # unit_vector_distance = get_color_distance(self.base_color_unit_vector, get_unit_vector(current_color))

        # if not self.is_pressed and angle > 0:
        #     self.on_press(self)
        #     self.is_pressed = True
        #     # print(color_distance, unit_vector_distance)
        #     # print(self.note + self.octave + " pressed")

        # elif self.is_pressed and get_color_distance(current_color, self.base_color) < self.press_threshold:
        #     self.on_release(self)
        #     self.is_pressed = False
        

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