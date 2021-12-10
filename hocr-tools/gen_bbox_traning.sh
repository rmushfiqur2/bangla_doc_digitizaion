#!/bin/bash
./page2hocr.sh
./crop-images.sh
./gen-box-file.sh
./edit-box-file.sh

echo "Congratulations! training data generation for character segmentation completed for first stage. Now run python3 labelImg.py to correct labels."
