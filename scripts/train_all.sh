#!/bin/bash
# export CUDA_VISIBLE_DEVICES="0"

train_models () {
  tile=$1
  batch=$2
  epoch=30
  dest="weights/gen3/$tile"

  models=(
    "unet16b"
    "unet16n"
    "unet16"
    "albunet"
    "albunet_n"
    "albunet_b"
    "unet11b"
    "unet11n"
    "unet11"
  )
  for model in "${models[@]}" ; do
    (set -x; python train.py \
      -m "${model}" -b $batch -e $epoch --tile $tile --dest $dest)
  done
}

train_models 768 4
train_models 512 8
train_models 256 32
