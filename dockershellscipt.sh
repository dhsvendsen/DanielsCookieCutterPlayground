#!/bin/bash
# exit when any command fails
set -e
dvc pull
python -u src/models/train_model.py train
