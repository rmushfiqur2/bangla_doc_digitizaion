#!/bin/bash
SOURCE="./myfiles/"
lang=ben_edt
set -- "$SOURCE"doc*.xml
for img_file; do
    echo -e  "\r\n File: $img_file"
    ocr-transform page hocr ${img_file} ${img_file%.*}.hocr
done

echo "hOCR files produced from PAGE xml files. Now extract images using ./crop-images.sh"
