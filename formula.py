import numpy as np


IDX_NONE, IDX_NORMAL, IDX_GLEASON_3, IDX_GLEASON_4, IDX_GLEASON_5 = range(5)
INDEX_MAP = np.array([
    IDX_NONE,      # empty
    IDX_NORMAL,    # 000: black
    IDX_GLEASON_3, # B00: blue
    IDX_GLEASON_4, # 0G0: green
    IDX_GLEASON_3, # BG0: cyan
    IDX_GLEASON_5, # 00R: red
    IDX_NORMAL,    # B0R: purple
    IDX_GLEASON_4, # 0GR: yellow
    IDX_NONE,      # BGR: white
])
NUM_CLASSES = len(np.unique(INDEX_MAP))

COLOR_MAP = np.array([
    [   0,   0,   0,   0], # 0 -> transparent
    [   0,   0,   0, 255], # 1 -> black
    [ 255,   0,   0, 255], # 2 -> blue
    [   0, 255,   0, 255], # 3 -> green
    [   0,   0, 255, 255], # 4 -> red
], dtype='uint8')

COLOR_MAP_ALPHA = np.array([
    [   0,   0,   0,   0], # 0 -> transparent
    [   0,   0,   0, 255], # 1 -> black
    [ 255,   0,   0, 255], # 2 -> blue
    [   0, 255,   0, 255], # 3 -> green
    [   0,   0, 255, 255], # 4 -> red
], dtype='uint8')
