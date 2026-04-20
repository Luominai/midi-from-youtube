import math
from typing import Sequence
import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from scipy import stats
from setup import setup_video_capture
import key as Key


class KeyboardParser3:
    def __init__(self):
        self.scan_progress = 0
        self.votes = {}
        self.vote_threshold = 90
        self.vote_verdict = None
        self.num_strata = 50
        self.batch = 5
        self.key_pattern = None
        self.keys = []


    def process(self, frame):
        paused = False
        # resize the frame to a consistent height of 720px
        frame = resize(frame)
        (height, width, channels) = frame.shape

        # if there is no verdict on what the keys are, scan the frame
        if self.vote_verdict is None:
            self.vote_verdict = scan(frame, self.batch, self.num_strata, self.scan_progress, self.vote_threshold, self.votes)
            self.scan_progress = (self.scan_progress + 1) % (self.num_strata // self.batch)

        # if there is a verdict and the keys are not yet defined, use the verdict to set the keys
        if self.vote_verdict is not None and len(self.keys) == 0:
            num_votes, layers = self.votes[self.vote_verdict]
            self.keys = get_keys(frame, self.votes, self.vote_verdict)

        if len(self.keys) != 0:
            for key in self.keys:
                key.process(frame)
                # if key.note == "D" and key.octave == 1:
                #     key.process(frame)


        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = edge_detection(frame)
        cv.imshow("frame", frame)
        return paused


def resize(frame):
    (height, width, channels) = frame.shape
    scale = 720 / height
    frame = cv.resize(frame, None, fx=scale, fy=scale)
    return frame


def print_key(key):
    print(key.note)


# returns an array of Key objects
def get_keys(frame, votes, vote_verdict):
    # figure out the order of keys in the layers
    key_pattern = get_pattern(votes, vote_verdict)

    # using the key pattern, label every piece of land with its note and octave
    num_votes, layers = votes[vote_verdict]
    for strata_num, layer in layers.items():
        [valleys, plateaus, full_survey] = layer
        label_terrain(full_survey, pattern=key_pattern)

    # sort the land into keys labeled by octave and note
    keys_by_octave_and_note = sort_layers(list(layers.values()))

    # create Key objects out of the sorted result
    keys = []
    for octave, key in enumerate(keys_by_octave_and_note):
        for note, strata in key.items():
            strata = list(map(tuple, strata))
            strata = np.array(strata, dtype=[
                ("start", "i2"),
                ("end", "i2"),
                ("y_pos", "i2"),
                ("is_valley", "bool"),
                ("note", "U2"),
                ("octave", "i1")
            ])

            keys.append(Key.Key(frame, strata, note, octave, 
                                lambda a: print("pressed", a.note + str(a.octave)), 
                                lambda a: print("released", a.note + str(a.octave))))

    return keys



def scan(frame: MatLike, batch_size: int, num_strata: int, batch_num: int, vote_threshold: int, votes: dict) -> int | None:
    """
    Scans a frame for keyboard keys and votes according to the number of detected white keys. \n
    This function splits the lower half of the frame into strata and performs detection on batches of them at a time. \n
    Returns the number of detected white keys if this iteration has made a verdict on where the keys are. Returns ```None``` otherwise.

    Parameters
    ----------
    frame : Matlike
        the image to scan
    batch_size : int
        how many strata to perform detection on at a time
    num_strata: int
        how many strata in total
    batch_num: int
        which batch to perform detection on
    vote_threshold: int
        how many votes are needed to make a verdict
    votes: dict
        A dictionary with the following structure \n
        ```
        <num_keys> : [ 
            <num_votes>, 
            { 
                <stratum_num>: [ <valleys>, <plateaus>, <full_survey> ] 
            } 
        ]
        ```
    """
    # limit the scan to the bottom half of the screen
    (height, width, channels) = frame.shape
    layers, positions = stratify(frame, num_strata, top=height//2)
    vote_verdict = None

    # batch the scans
    for i in range(batch_size * batch_num, batch_size * (batch_num + 1)):
        layer = layers[i]
        y_pos = positions[i]
        binary = binarize(layer)
        valleys, plateaus, full_survey = get_terrain(binary, y_pos)
        # valleys, plateaus, full_survey = adaptive_quantization(layer, int(y_pos))
        valid = is_valid(plateaus, valleys, full_survey)

        if valid:
            num_votes = cast_vote(valleys, plateaus, full_survey, i, votes)
            print("strata", str(i), ":", len(plateaus), num_votes)

            if num_votes >= vote_threshold and has_pattern(full_survey):
                vote_verdict = len(plateaus)
                print("vote decided for", vote_verdict)
                break

        draw_terrain(frame, plateaus, color = (0,255,0) if valid else (0,0,255))
        draw_terrain(frame, valleys, color = (255,0,0) if valid else (0,0,255))

    return vote_verdict



def get_pattern(votes, vote_verdict) -> Sequence[str] | None:
    """
    Iterates over the stratum with the lowest y-value (meaning higher up on the image) to determine the order of white notes for all strata \n
    Returns an array of strings representing the notes of the first 7 white keys in order. Returns ```None``` if no pattern was detected

    Parameters
    ----------
    votes: dict
        A dictionary with the following structure \n
        ```
        <num_keys> : [ 
            <num_votes>, 
            { 
                <stratum_num>: [ <valleys>, <plateaus>, <full_survey> ] 
            } 
        ]
        ```
    vote_verdict: int
        the ```<num_keys>``` to index ```votes``` with
    """
    notes = "C D E F G A B".split(" ")

    [num_votes, strata] = votes[vote_verdict]
    strata_nums = list(strata.keys())
    strata_nums.sort()
    [valleys, plateaus, full_survey] = strata[strata_nums[0]]

    index_of_pattern = find_pattern(plateaus)
    
    # print(index_of_pattern)
    if index_of_pattern == -1:
        return None
    order = notes[index_of_pattern % len(notes):] + notes[:index_of_pattern % len(notes)]
    print(order)
    return order
    


def draw_terrain(frame, terrain, color, size=3, draw_text=True):
    """
    Draws rectangles representing the start and end positions of every tuple in the terrain. \n
    Rectangles have inflated heights for visibility. \n
    If ```pattern``` is given, this will also label all the keys

    Parameters
    ----------
    frame : Matlike
        the image to draw on
    terrain: Sequence( list( int, int, bool, str, int ) )
        a list of lists with the following structure
        ```
        [ 
            [ <start>, <end>, <is_valley>, <note>, <octave> ], 
            ... 
        ]
        ```
    y_pos: int
        the y value of all pixels in ```terrain```
    color: Tuple( int, int, int )
        color of the line visualizing the stratum (not ```terrain``` itself)
    pattern: Sequence( str ) = ```None```
        a list of strings representing the notes of the first 7 white keys in order
    """
    octave_colors = [(180,0,0),(0,180,0),(0,0,180),(180,180,0),(0,180,180),(180,0,180)]

    for idx, (start, end, y_pos, is_valley, note, octave) in enumerate(terrain):
        if draw_text and note != "?" and octave > -1:
            octave_color = octave_colors[octave % len(octave_colors)]
            octave_color = (octave_color[0] + 40, octave_color[1] + 40, octave_color[2] + 40)
            cv.putText(frame, note, (start, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, octave_color, 1)

        if color is not None:
            cv.rectangle(frame, (start, y_pos - size), (end, y_pos + size), color, 1)

            # if idx == len(terrain) - 1:
            #     cv.rectangle(frame, (0, y_pos), (frame.shape[1], y_pos), (100,0,0), 1) # type: ignore



def label_terrain(terrain, pattern):
    plateau_index = 0
    current_octave = 0

    # (0: start, 1: end, 2: y_pos, 3: is_valley, 4: note, 5: octave) = terrain
    for idx, ground in enumerate(terrain):
        in_valley = ground[3]
        note = pattern[plateau_index % len(pattern)]
        note = pattern[(plateau_index - 1) % len(pattern)] + "#" if in_valley else note

        if note == "C" and idx > 0:
            current_octave += 1

        ground[4] = note
        ground[5] = current_octave

        if not in_valley:
            plateau_index += 1



def sort_layers(layers):
    """
    [
        {
            "A": [ [start, end, ...], ... ],
            "A#": [ [start, end, ...], ... ],
            ...
        }
    ]
    The element at index i belongs to octave i
    """
    terrain_by_octave_and_note = []

    for [valleys, plateaus, full_survey] in layers:
        for idx, (start, end, y_pos, is_valley, note, octave) in enumerate(full_survey):
            # ensure we have a number of Dictionaries equal to number of octaves
            if len(terrain_by_octave_and_note) <= octave:
                for i in range(octave - len(terrain_by_octave_and_note) + 1):
                    terrain_by_octave_and_note.append({})

            # add an array for every note within an octave
            if note not in terrain_by_octave_and_note[octave]:
                terrain_by_octave_and_note[octave][note] = []

            # add a piece of land under its given octave and note
            terrain_by_octave_and_note[octave][note].append(full_survey[idx])

    return terrain_by_octave_and_note



def stratify(frame, max_layers, top = 0, offset = 0, limit = None, reverse = False):
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


def binarize(stratum):
    # the data is in the incorrect format for some reason so we fix it with this line
    stratum = np.uint16(stratum)
    stratum = cv.cvtColor(stratum, cv.COLOR_BGR2GRAY) # type: ignore
    thresh, otsu = cv.threshold(stratum, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return otsu


# [ <start>, <end>, <y_pos>, <is_valley>, <note> = "?", octave> = -1 ]
def get_terrain(binary_stratum, y_pos, min_plat_size = 6, min_valley_size = 6):
    values = binary_stratum.flatten()
    terrain = []
    valleys = []
    plateaus = []
    start = 0

    for i, curr in enumerate(values):
        # if this is the first element, we can treat prev as curr so we end up ignoring it
        prev = values[i - 1] if i > 0 else curr
        
        # if this pixel is the same as the previous, ignore
        if curr == prev:
            continue

        # when the pixel value changes, record the land between now and the previous change. the land is a valley if its pixels are black
        is_valley = prev == 0
        meets_min_size = (is_valley and i - start >= min_valley_size) or (not is_valley and i - start >=min_plat_size)
        
        # if there isn't enough land, we'll skip over this section
        if meets_min_size:
            land = [start, i, y_pos, is_valley, "?", -1]
            terrain.append(land)
            valleys.append(land) if is_valley else plateaus.append(land)

        # mark this pixel as the start of new land
        start = i

    # record the last bit of land that wasn't saved in the for loop
    is_valley = values[start] == 0
    meets_min_size = (is_valley and len(values) - start >= min_valley_size) or (not is_valley and len(values) - start >= min_plat_size)
    if meets_min_size:
        land = [start, len(values), y_pos, is_valley, "?", -1]
        terrain.append(land)
        valleys.append(land) if is_valley else plateaus.append(land)

    return (valleys, plateaus, terrain)



# def quantize_colors(frame, num_colors):
#     thresh, otsu = cv.threshold(frame, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     colors = [0, 255]
#     best_labels = otsu.flatten()

#     # # take the frame and reshape it into a 1d array of BGR tuples
#     # data = np.float32(frame).reshape((frame.shape[0] * frame.shape[1], 3))

#     # # define criteria, number of clusters(K) and apply kmeans()
#     # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
#     # retval, best_labels, centers = cv.kmeans(
#     #     data=data,  # type: ignore
#     #     K=num_colors,
#     #     bestLabels=None, # type: ignore
#     #     criteria=criteria, 
#     #     attempts=5, 
#     #     flags=cv.KMEANS_RANDOM_CENTERS
#     # )

#     # colors = np.uint8(centers)

#     return (colors, best_labels)



def adaptive_quantization(stratum, y_pos):
    
    binary = binarize(stratum)
    valleys, plateaus, full_survey = get_terrain(binary, y_pos)

    # # print("==================================")
    # for i in range(2,5):
    #     # uniform = is_uniform(plateaus) 
    #     # pattern = has_pattern(full_survey)
    #     valid = is_valid(plateaus, valleys, full_survey)
    #     # print(y_pos, "k", i, "is_uniform:", uniform, "has_pattern:", pattern, "valid", valid)

    #     if not valid:
    #         (colors, labels) = quantize_colors(stratum, i)
    #         valleys, plateaus, full_survey = get_terrain(labels, y_pos)
    #     else:
    #         break

    return (valleys, plateaus, full_survey)



def is_valid(plateaus, valleys, full_survey):
    # arbitrary minimum but I don't see any piano tutorials with less than a full octave
    # has_min_key_count = len(plateaus) >= 7
    # if not has_min_key_count:
    #     return False
    
    # check if the terrain has a keyboard pattern
    if not has_pattern(full_survey):
        return False

    # # check if the terrain has roughly similar size
    # if not is_uniform(plateaus):
    #     return False

    return True



def find_pattern(plateaus, gap_thresh = 4):
    """
    Returns the index of the first octave starting at C. Used for labeling a known keyboard
    """
    keyboard = "101011010101" + "101011010101"
    pattern = ""

    if (len(plateaus) < 8):
        return -1

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



def has_pattern(full_survey):
    """
    Checks if the plateaus and valleys of the terrain are arranged in a keyboard pattern. Use to help check if a keyset is a keyboard
    """

    if (len(full_survey) < 12):
        return False
    
    # # if all plateaus and no valleys, it is a valid pattern only if the plateaus are also evenly spaced
    # if all([not is_valley for (start, end, y_pos, is_valley, *rest) in full_survey]):
    #     return is_uniform(full_survey)
    
    keyboard = "101011010101" + "101011010101"
    pattern = ""

    for i in range(0, 12):
        (start, end, y_pos, is_valley, *rest) = full_survey[i]
        if is_valley:
            pattern += "0"
        else:
            pattern += "1"

    index_of_pattern = keyboard.find(pattern)
    if index_of_pattern == -1:
        return False
    
    for i in range(12, len(full_survey)):
        (start, end, y_pos, is_valley, *rest) = full_survey[i]
        expected = "0" if is_valley else "1"
        if keyboard[(index_of_pattern + i) % len(keyboard)] != expected:
            return False
        
    return True



def is_uniform(terrain, buffer = 1, scale_thresh = 1.5, pixel_thresh = 8):
    """
    Checks if all terrain is siilar in width
    """
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



# def get_terrain(labels, y_pos, min_plat_size = 6, min_valley_size = 6):
#     """
#     Given a list of labels, returns 3 lists of tuples. \n
#     The first 2 lists represent valley and plateau features. The third contains tuples from both features in order. \n
#     A pixel is considered plateau if it is labeled with the most common label \n
#     A pixel is considered valley if it is labeled with any label other than the most common \n
    
#     Each tuple has the following structure
#     ```
#     [ <start>, <end>, <y_pos>, <is_valley>, <note> = "?", octave> = -1 ]
#     ```
#     """
#     terrain = labels.flatten()
#     plateau_label, _ = stats.mode(terrain)
#     in_valley = False

#     valleys = []
#     plateaus = []
#     full_survey = []
#     start_of_valley = 0
#     start_of_plateau = 0
    
#     for i in range(len(labels)):
#         if not in_valley and terrain[i] != plateau_label:
#             in_valley = True
#             start_of_valley = i
#             if (i - start_of_plateau) >= min_plat_size:
#                 entry = [start_of_plateau, i, y_pos, False, "?", -1]
#                 plateaus.append(entry)
#                 full_survey.append(entry)

#         elif in_valley and terrain[i] == plateau_label:
#             in_valley = False
#             start_of_plateau = i
#             if (i - start_of_valley) >= min_valley_size:
#                 entry = [start_of_valley, i, y_pos, True, "?", -1]
#                 valleys.append(entry)
#                 full_survey.append(entry)

#     if in_valley and (len(terrain) - start_of_valley) >= min_valley_size:
#         entry = [start_of_valley, len(terrain), y_pos, True, "?", -1]
#         valleys.append(entry)
#         full_survey.append(entry)
#     elif not in_valley and (len(terrain) - start_of_plateau) >= min_plat_size:
#         entry = [start_of_plateau, len(terrain), y_pos, False, "?", -1]
#         plateaus.append(entry)
#         full_survey.append(entry)

#     return valleys, plateaus, full_survey



# first indexed by plateau count, then by strata number
def cast_vote(valleys, plateaus, full_survey, strata_num, votes):
    choice = len(plateaus)

    # when we count a new number of keys, file it under history
    if choice not in votes.keys():
        votes[choice] = [1, {
            strata_num: [valleys, plateaus, full_survey]
        }]

    # when we count a number of keys we've seen before, update history
    else:
        num_votes, strata_data = votes[choice]
        votes[choice][0] = num_votes + 1
        strata_data[strata_num] = [valleys, plateaus, full_survey]

    return (votes[choice][0])


