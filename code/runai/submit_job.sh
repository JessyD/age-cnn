#!/bin/bash
runai submit ab --image nvcr.io/nvidia/pytorch:20.03-py3 \
  --gpu 1 \
  --project wds20 \
  --command bash --args /project/code/runai/train_banc.sh \
  --large-shm \
  --volume /nfs/home/wds20/projects/age-cnn:/project
