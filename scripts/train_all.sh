#!/bin/bash
# export CUDA_VISIBLE_DEVICES="0"

train_models () {
  tile=$1
  batch=$2
  epoch=30
  dest="weights/gen3/$tile"

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
    (set -x; echo python train.py \
      -m "${model}" -b $batch -e $epoch --tile $tile --dest $dest)
  done
}

train_models 256 32
train_models 512 8
train_models 768 4
