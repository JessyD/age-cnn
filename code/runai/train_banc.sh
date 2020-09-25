#!/bin/bash

printf '%s\n' --------------------
echo PIP
printf '%s\n' --------------------
pip install -r /project/requirements.txt -q
pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
echo SUCCESS

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
python3 /project/code/train.py