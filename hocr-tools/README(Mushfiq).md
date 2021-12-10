### Important files:

** ALl shell files are genrated inspired by this linked solution from 

```NOTE: If you want to generate line images for transcription from a full page, see tips in issue 7 and in particular @Shreeshrii's shell script.""" Deatils: https://github.com/tesseract-ocr/tesstrain
```
[doc2croppedlines.sh] -> The solution of @Shreeshrii has been adapted as doc2croppedlines.sh
[hocr-extract-images] -> (python) the code for cropping and estimating gt (as plain text) for each lines.
[create_ml_io.py] -> adapted form [~/Documents/labelImg] to support createMl (json) annotation format (coco format) by [edit-box-file]

### Newly created files:

[page2hocr.sh] -> uses built cmd from [~/Documents/ocr-fileformat]
[crop-images] -> python adapted from [hocr-extract-images]
[crop-images.sh] -> from images and hocr files create small images
[gen-box-file.sh] -> shell file for creating box files for each lines (via tesseract). Then it will be corrected using [~/Documents/labelImg]
[edit-box-file] -> tessract box to yolo labels/other label format
[edit-box-file.sh] -> corresponding shell file for edit-box-file


[gen_bbox_traning.sh] -> all the above commands one after another

