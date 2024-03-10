#!/bin/bash

python image_train.py --data_dir $DATASET_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
