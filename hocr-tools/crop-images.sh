#!/bin/bash
SOURCE="./myfiles/"
lang=ben_edt
set -- "$SOURCE"doc*.jpg
for img_file; do
    echo -e  "\r\n File: $img_file"
    PYTHONIOENCODING=UTF-8 ./crop_images -b ./myfiles/ -p "${img_file%.*}"-%03d.jpg -s cropped -img "${img_file}" "${img_file%.*}".hocr 
done

echo "Images cropped according to given hOCR files. Now run ./gen-box-file.sh to generate box file from cropped images"
