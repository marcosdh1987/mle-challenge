#!/bin/bash
DIR=datasets/kaggle_dataset2/
mkdir -p $DIR
wget -P $DIR https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip
cd $DIR
unzip dataset-resized.zip
mv dataset-resized images

# Clean up
rm dataset-resized.zip
rm -rf __MACOSX/

echo "Goodbye!"
