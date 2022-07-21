#!/bin/sh

cd data
save_folder="./tiny-imagenet-200/"
if [ -d "$save_folder" ]
then
    echo "Already got the tiny imagenet dataset!"
    exit 0
else
    wget https://raw.githubusercontent.com/yandexdataschool/Practical_DL/spring2019/week03_convnets/tiny_img.py -O tiny_img.py
    python -c "import tiny_img; tiny_img.download_tinyImg200('.')"