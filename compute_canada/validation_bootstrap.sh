#!/bin/bash


script_dir="/home/cderoy/sex_diff/"

source /home/cderoy/elmvenv/bin/activate
python3 ${script_dir}/val_bootstrap_discriminant_features.py 
