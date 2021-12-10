#!/bin/bash
SOURCE="./myfiles/cropped/"
lang=ben_edt
set -- "$SOURCE"*.jpg
for img_file; do
    echo -e  "\r\n File: $img_file"
    OMP_THREAD_LIMIT=1 tesseract "${img_file}" "${img_file%.*}"  --psm 12  -l $lang -c page_separator='' -c hocr_char_boxes=1 makebox
done

echo "For each cropped lines, bounding box of glyphs has been estimated using tesseract(.box). Now convert them to compatible format (i.e. yolo) using ./edit-box-file.sh"
