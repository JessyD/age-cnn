#!/bin/bash
#Training
runai submit ab --image nvcr.io/nvidia/pytorch:20.03-py3 \
  --gpu 1 \
  --project wds20 \
  --command bash --args /project/code/runai/train_banc.sh \
  --large-shm \
  --volume /nfs/home/wds20/projects/age-cnn:/project

# Predict BANC test
runai submit pred_ab --image nvcr.io/nvidia/pytorch:20.03-py3 \
  --gpu 1 \
  --project wds20 \
  --command bash --args /project/code/runai/predict_banc.sh \
  --large-shm \
  --volume /nfs/home/wds20/projects/age-cnn:/project

# Debug
runai submit entrada --image nvcr.io/nvidia/pytorch:20.03-py3 \
--gpu 1 \
--command sleep --args 10000 -p wds20 --large-shm --memory 64G --volume /nfs/home/wds20/projects/age-cnn:/project
