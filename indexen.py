import sys
import os
import gc
from PIL import Image, ImageFilter

if len(sys.argv) < 3:
    print('Invalid arguments.')
    exit(1)

OUTPUT_DIR = 'train'

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

print(INPUT_DIR, OUTPUT_DIR)
exit(0)


file_names = os.listdir(INPUT_DIR)
for file_name in file_names:
    img = Image.open(file_name)

    yy = []
    xx = []
    names = []
    for case_name in case_names:
        case_dir = f'{base_dir}/{case_name}'
        file_names = os.listdir(f'{case_dir}/y/')
        file_names.sort()
        i = 0
        count = 0
        for file_name in file_names:
