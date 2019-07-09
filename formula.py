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
    IDX_GLEASON_5, # 0GR: yellow
    IDX_NONE,      # BGR: white
])
NUM_CLASSES = len(np.unique(INDEX_MAP))
