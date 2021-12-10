#!/bin/bash
SOURCE="./myfiles/cropped/"
set -- "$SOURCE"*.box
for box_file; do
    echo -e  "\r\n File: $box_file"
    PYTHONIOENCODING=UTF-8 ./edit-box-file -b ./myfiles/cropped/ -p "${box_file%.*}" -img jpg -annot coco "${box_file}"
done
#echo "glyph" > ./myfiles/cropped/classes.txt # not required for coco

echo "Box file converted to yolo/coco labels (.txt/.json). Now edit this yolo labels in a user interface (i.e. labelImg)"
