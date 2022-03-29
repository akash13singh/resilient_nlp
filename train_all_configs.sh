#!/bin/sh

cd configs
CONFIGS=`echo *.json`
cd ..

for i in $CONFIGS
do
    python -u train_using_config.py configs/$i output/${i/\.json/.pth} | tee output/${i/\.json/.log}
done
