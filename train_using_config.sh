#!/bin/bash

python -u train_using_config.py configs/$1 output/${1/\.json/.pth} | tee output/${1/\.json/.log}
