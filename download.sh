"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

FILE=$1

if [ $FILE == "pretrained-network-k-hairstyle" ]; then
    URL=
    ZIP_FILE=./expr/checkpoints/k-hairstyle.zip
    mkdir -p ./expr/checkpoints
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./expr/checkpoints
    rm $ZIP_FILE
    
elif  [ $FILE == "k-hairstyle-dataset" ]; then
    URL=
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE
    mkdir -p ./expr/checkpoints
    OUT_FILE=./expr/checkpoints/60000_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE
    
else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi
