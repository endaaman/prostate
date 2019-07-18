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

sizes=(256 512 768)

for size in "${sizes[@]}" ; do
  for model in "${models[@]}" ; do
    (set -x; python validate.py \
      -m "${model}" \
      -w "weights/gen2/${size}/${model}/${model}_30.pt" \
      --dest "report/gen3/${size}" \
      --cpu --size 3000 \
      )
  done
done
