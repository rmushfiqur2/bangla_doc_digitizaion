#!/usr/bin/env python

# change box files generated with tesseracr (makebox) to yolo/ other format

from __future__ import print_function
import argparse
import codecs
import os
import re
import sys

from PIL import Image
from create_ml_io import CreateMLWriter



parser = argparse.ArgumentParser(
    description=("Convert the box file (tesseract makebox) to compatible format "
                 "designed for segmentation training (box file-> labels)")
)
parser.add_argument(
    "file",
    help="box file",
    nargs='?',
    default="")
parser.add_argument("-b", "--basename", help="boxfiles-dir")
parser.add_argument(
    "-p",
    "--pattern",
    help="name-pattern of label file without ext, default: %(default)s",
    default="annotation")
parser.add_argument(
    "-img",
    "--image_format",
    help="image file type associated with (for size calculation), default: %(default)s",
    default="tif")
parser.add_argument(
    "-annot",
    "--out_annotate_format",
    help="output annonation format (coco/yolo), default: %(default)s",
    default="yolo")
"""parser.add_argument(
    "-P",
    "--pad",
    default=None,
    help="extra padding for bounding box")"""
"""parser.add_argument(
    "-U",
    "--unicodedammit",
    action="store_true",
    help="attempt to use BeautifulSoup.UnicodeDammit to fix encoding issues")"""
args = parser.parse_args()

bname = args.file # input box file
tpattern = args.pattern # output file name (without ext)
annot_type = args.out_annotate_format

if annot_type == "yolo":
    tpattern = tpattern + '.txt'
elif annot_type == "coco":
    tpattern = tpattern + '.json'
else:
    print("Unsupported annotation type:", annot_type)
    sys.exit(1)

if args.basename:
    bname = os.path.join(args.basename, os.path.basename(bname))
if not os.path.exists(bname):
    print("box file not found:", bname)
    sys.exit(1)

iname = bname[:-3] + args.image_format
image = Image.open(iname)
iname_dir, iname_base = os.path.split(iname)
width, height = image.size

with open(bname) as f:
    lines = f.readlines()
    if annot_type == "yolo":
        f = codecs.open(tpattern, 'w', 'utf-8')
        for line in lines:
            #print(line)
            #print(line.split())
            #unichar, xmin, ymin, xmax, ymax, page_no = line.split()
            bbox = [int(x) for x in line.split()[1:5]] #x1y1x2y2
            bbox[1] = height-1-bbox[1] # y is 0 at bottom (tesseract)
            bbox[3] = height-1-bbox[3] # y is 0 at bottom (tesseract)
            # x1y2x2y1
            #print(bbox)
            xmin = bbox[0]/float(width)
            xmax = bbox[2]/float(width)
            ymin = bbox[3]/float(height)
            ymax = bbox[1]/float(height)
            bbox_float = [xmin, ymin, xmax, ymax]
            x = (xmin+xmax)/2
            y = (ymin+ymax)/2
            w = xmax - xmin
            h = ymax - ymin
            xywh = [x, y, w, h] # yolo
            #coords = ' '.join([str(coord) for coord in bbox])
            coords = ' '.join(["{:.4f}".format(coord) for coord in xywh])
            f.write('0 '+coords+'\n')
        f.close
    elif annot_type == "coco":
        shapes = []
        for line in lines:
            bbox = [float(x) for x in line.split()[1:5]] #x1y1x2y2
            bbox[1] = height-1-bbox[1] # y is 0 at bottom (tesseract)
            bbox[3] = height-1-bbox[3] # y is 0 at bottom (tesseract)
            # x1y2x2y1
            #print(bbox)
            shapes.append({"points": bbox, "label": "glyph"}) # create_ml_io resolves x1y2x2y1
            # coco is also xywh format
        cocoWriter = CreateMLWriter("", iname_base, [width, height], shapes, tpattern, database_src='NBR')
        cocoWriter.write()

