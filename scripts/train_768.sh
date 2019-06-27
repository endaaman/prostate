#!/bin/bash
# export CUDA_VISIBLE_DEVICES="0"

epoch=30
tile=768
batch=4
dest="weights/$tile"

models=(
  "unet11"
  "unet11b"
  "unet11n"
  "unet16"
  "unet16b"
  "unet16n"
  "albunet"
  "albunet_b"
  "albunet_n"
)

for model in "${models[@]}" ; do
  python train.py -m "${model}" -b $batch -e $epoch --tile $tile --dest $dest
done
