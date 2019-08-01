#!/bin/bash

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

sizes=(768)

for size in "${sizes[@]}" ; do
  for model in "${models[@]}" ; do
    dest="report/gen3_val/${size}"
    (set -x; python validate.py \
      -m "${model}" \
      -w "weights/gen3/${size}/${model}/${model}_30.pt" \
      --dest $dest \
      --cpu --size 3000 \
      --target val
      )
  done
done
