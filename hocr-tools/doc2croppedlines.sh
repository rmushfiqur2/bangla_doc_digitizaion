#!/bin/bash
SOURCE="./myfiles/"
lang=ben_edt
set -- "$SOURCE"*.tif
for img_file; do
    echo -e  "\r\n File: $img_file"
    OMP_THREAD_LIMIT=1 tesseract "${img_file}" "${img_file%.*}"  --psm 12  -l $lang -c page_separator='' -c hocr_char_boxes=1 makebox
    #source venv/bin/activate
    PYTHONIOENCODING=UTF-8 ./hocr-extract-images -b ./myfiles/ -p "${img_file%.*}"/%03d.exp0.tif  "${img_file%.*}".hocr 
    #deactivate
done
#rename s/exp0.txt/exp0.gt.txt/ ./myfiles/*exp0.txt

echo "Congratulations! 1st stage training data generation completed. Now you may want to run cd ~/Documents/labelImg and then python3 labelImg.py to correct labels."
