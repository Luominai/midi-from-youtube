import math
import cv2 as cv
import numpy as np
from scipy import stats
from setup import setup_video_capture

class KeyboardParser:
    def __init__(self):
        self.scan_progress = 0
        self.votes = {}
        self.vote_threshold = 60
        self.vote_verdict = None
        self.num_strata = 25
        self.batch = 5
        self.key_pattern = None


    def process(self, frame):
        paused = False
        (height, width, channels) = frame.shape
        scale = 720 / height
        frame = cv.resize(frame, None, fx=scale, fy=scale)
        (height, width, channels) = frame.shape


        if self.vote_verdict is None:
            self.scan(frame)

        if self.vote_verdict is not None:
            if self.key_pattern is None:
                self.key_pattern = self.get_pattern()
        
            num_votes, layers = self.votes[self.vote_verdict]

            for strata_num, layer in layers.items():
                [valleys, plateaus, full_survey] = layer

                step = (height // 2) / (self.num_strata + 1)
                y_pos = math.floor(step * (strata_num + 1)) + (height // 2)
                self.draw_terrain(frame, full_survey, y_pos, (255,0,0), pattern=self.key_pattern)

        cv.imshow("frame", frame)
        return paused


    
    def scan(self, frame):
        (height, width, channels) = frame.shape
        layers, positions = self.stratify(frame, self.num_strata, top=height//2)

        for i in range(self.batch * self.scan_progress, self.batch * (self.scan_progress + 1)):
            layer = layers[i]
            valleys, plateaus, full_survey = self.adaptive_quantization(layer)
            # plateaus, valleys, full_survey = adaptive_quantization(layer)
            y_pos = positions[i]
            valid = self.is_valid(plateaus, valleys)

            self.draw_terrain(frame, plateaus, y_pos, color = (0,255,0) if valid else (0,0,255))
            self.draw_terrain(frame, valleys, y_pos, color = (255,0,0) if valid else (0,0,255))

            if valid:
                num_votes = self.cast_vote(valleys, plateaus, full_survey, i)
                print("strata", str(i), ":", len(plateaus), num_votes)

                if num_votes >= self.vote_threshold and self.find_pattern(plateaus) != -1:
                    self.vote_verdict = len(plateaus)
                    print("vote decided for", self.vote_verdict)
                    break

        self.scan_progress = (self.scan_progress + 1) % (len(layers) // self.batch)


    def get_pattern(self):
        notes = "C D E F G A B".split(" ")

        [num_votes, strata] = self.votes[self.vote_verdict]
        strata_nums = list(strata.keys())
        strata_nums.sort()
        [valleys, plateaus, full_survey] = strata[strata_nums[0]]

        index_of_pattern = self.find_pattern(plateaus)
        
        print(index_of_pattern)
        if index_of_pattern == -1:
            return None
        order = notes[index_of_pattern % len(notes):] + notes[:index_of_pattern % len(notes)]
        print(order)
        return order
        

    def draw_terrain(self, frame, terrain, y_pos, color, pattern=None):
        plateau_index = 0
        current_octave = 0
        octave_colors = [(180,0,0),(0,180,0),(0,0,180),(180,180,0),(0,180,180),(180,0,180),]

        for idx, (start, end, is_valley, note, octave) in enumerate(terrain):
            if pattern is not None:
                note = pattern[plateau_index % len(pattern)]

                if note == "C" and idx > 0:
                    current_octave += 1

                octave = current_octave
                octave_color = octave_colors[octave % len(octave_colors)]
        
                if is_valley:
                    note = pattern[(plateau_index - 1) % len(pattern)] + "#"
                    octave_color = (octave_color[0] + 40, octave_color[1] + 40, octave_color[2] + 40)
                else:
                    plateau_index += 1

                cv.putText(frame, note, (start, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, octave_color, 1)
                continue
                
            cv.rectangle(frame, (start, y_pos - 5), (end, y_pos + 5), color, 2)

        if pattern is None:
            cv.rectangle(frame, (0, y_pos), (frame.shape[1], y_pos), (100,0,0), 1) # type: ignore


    def stratify(self, frame, max_layers, top = 0, offset = 0, limit = None, reverse = False):
        limit = max_layers if limit is None else limit
        (height, width, channels) = frame.shape
        layers = np.empty(shape=(limit, 1, frame.shape[1], 3))
        positions = np.empty(shape=limit, dtype=int)
        step = (height - top) / (max_layers + 1)

        for i in range(limit):
            step_num = i + max_layers - limit + 1 if reverse else i + 1
            y_pos = math.floor(step * step_num) + top + offset
            positions[i] = y_pos
            layers[i] = frame[y_pos]

        return layers, positions


    def quantize_colors(self, frame, num_colors):
        # take the frame and reshape it into a 1d array of BGR tuples
        data = np.float32(frame).reshape((frame.shape[0] * frame.shape[1], 3))

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        retval, best_labels, centers = cv.kmeans(
            data=data,  # type: ignore
            K=num_colors,
            bestLabels=None, # type: ignore
            criteria=criteria, 
            attempts=5, 
            flags=cv.KMEANS_RANDOM_CENTERS
        )

        colors = np.uint8(centers)

        return (colors, best_labels)


    def adaptive_quantization(self, frame):
        (colors, labels) = self.quantize_colors(frame, 3)
        valleys, plateaus, full_survey = self.get_terrain(labels)

        # if plateau are not uniform, retry using a different k
        if not self.is_uniform(plateaus):
            (colors, labels) = self.quantize_colors(frame, 2)
            valleys, plateaus, full_survey = self.get_terrain(labels)

        if not self.is_uniform(plateaus):
            (colors, labels) = self.quantize_colors(frame, 4)
            valleys, plateaus, full_survey = self.get_terrain(labels)

        return (valleys, plateaus, full_survey)


    def is_valid(self, plateaus, valleys):
        has_min_key_count = len(plateaus) >= 7
        if not has_min_key_count:
            return False
        
        if not self.is_uniform(plateaus) or not self.is_uniform(valleys):
            return False

        return True


    def find_pattern(self, plateaus, gap_thresh = 4):
        keyboard = "101011010101" + "101011010101"

        pattern = ""

        for i in range(0, 7):
            (curr_start, curr_end, *rest) = plateaus[i]
            (next_start, next_end, *rest) = plateaus[i + 1]

            if (next_start - curr_end > gap_thresh):
                pattern += "10"
            else:
                pattern += "1"

        # print(pattern)
        index_of_pattern = keyboard.find(pattern)
        # print(index_of_pattern)
        offset = -1 if index_of_pattern == -1 else sum([int(val) for val in keyboard[:index_of_pattern]])
        # print(offset)
        return offset


    def is_uniform(self, terrain, buffer = 1, scale_thresh = 1.5, pixel_thresh = 8):
        shortest = math.inf
        longest = 0.0

        # # we don't know the full extent of the first and last runs, so we exclude them from consideration
        for i in range(buffer, len(terrain) - buffer):
            (start, end, *rest) = terrain[i]
            dist = end - start

            if dist > longest:
                longest = dist

            if dist < shortest:
                shortest = dist

        if (longest / shortest) >= scale_thresh and (longest - shortest) > pixel_thresh:
            return False
        
        return True


    def get_terrain(self, labels, min_plat_size = 6, min_valley_size = 6):
        terrain = labels.flatten()
        plateau_label, _ = stats.mode(terrain)
        in_valley = False

        valleys = []
        plateaus = []
        full_survey = []
        start_of_valley = 0
        start_of_plateau = 0
        
        for i in range(len(labels)):
            if not in_valley and terrain[i] != plateau_label:
                in_valley = True
                start_of_valley = i
                if (i - start_of_plateau) >= min_plat_size:
                    entry = [start_of_plateau, i, False, "?", -1]
                    plateaus.append(entry)
                    full_survey.append(entry)

            elif in_valley and terrain[i] == plateau_label:
                in_valley = False
                start_of_plateau = i
                if (i - start_of_valley) >= min_valley_size:
                    entry = [start_of_valley, i, True, "?", -1]
                    valleys.append(entry)
                    full_survey.append(entry)

        if in_valley and (len(terrain) - start_of_valley) >= min_valley_size:
            entry = [start_of_valley, len(terrain), True, "?", -1]
            valleys.append(entry)
            full_survey.append(entry)
        elif not in_valley and (len(terrain) - start_of_plateau) >= min_plat_size:
            entry = [start_of_plateau, len(terrain), False, "?", -1]
            plateaus.append(entry)
            full_survey.append(entry)

        return valleys, plateaus, full_survey


    # first indexed by plateau count, then by strata number
    def cast_vote(self, valleys, plateaus, full_survey, strata_num):
        choice = len(plateaus)

        # when we count a new number of keys, file it under history
        if choice not in self.votes.keys():
            self.votes[choice] = [1, {
                strata_num: [valleys, plateaus, full_survey]
            }]

        # when we count a number of keys we've seen before, update history
        else:
            num_votes, strata_data = self.votes[choice]
            self.votes[choice][0] = num_votes + 1
            strata_data[strata_num] = [valleys, plateaus, full_survey]

        return (self.votes[choice][0])


    # setup_video_capture(process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")
    # setup_video_capture(process, "videos/Menu (from Kirby Air Riders) - Piano Tutorial [iElUjQXQkPc].webm")
    # setup_video_capture(process, "videos/Van Gogh by Virginio Aiello, On Piano - [Piano Tutorial] (Synthesia - SeeMusic) [2ESlH-fwxIc].webm")
    # setup_video_capture(process, "videos/BIRDBRAIN ｜ Jamie Paige PIANO TUTORIAL SHEET + MIDI [59qdAsKqIjA].webm")