#!/usr/bin/env python

# extract the images and texts within all the ocr_line elements
# within the hOCR file

from __future__ import print_function
import argparse
import codecs
import os
import re
import sys

from lxml import html
from PIL import Image

def get_prop(node, name):
    title = node.get("title")
    props = title.split(';')
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None

parser = argparse.ArgumentParser(
    description=("Extract the images using bbox information "
                 "elements within the hOCR file")
)
parser.add_argument(
    "file",
    help="hOCR file",
    type=argparse.FileType('r'),
    nargs='?',
    default=sys.stdin)
parser.add_argument("-b", "--basename", help="image-dir")
parser.add_argument(
    "-p",
    "--pattern",
    help="file-pattern, default: %(default)s",
    default="line-%03d.png")
parser.add_argument(
    "-s",
    "--subfolder",
    help="produce outfut to a subfolder, default: %(default)s",
    default="")
parser.add_argument(
    "-e",
    "--element",
    help="element-name, default: %(default)s",
    default="ocr_line")
parser.add_argument(
    "-img",
    "--imagesrc",
    help="image associated with the hOCR file, default: %(default)s",
    default="")
parser.add_argument(
    "-P",
    "--pad",
    default=None,
    help="extra padding for bounding box")
parser.add_argument(
    "-U",
    "--unicodedammit",
    action="store_true",
    help="attempt to use BeautifulSoup.UnicodeDammit to fix encoding issues")
args = parser.parse_args()

padding = None
if args.pad is not None:
    padding = eval("["+args.pad+"]")
    assert len(padding) in [1, 4], (args.pad, padding)
    if len(padding) == 1:
        padding = padding * 4

if args.unicodedammit:
    from bs4 import UnicodeDammit
    content = args.file.read()
    doc = UnicodeDammit(content, is_html=True)
    parser = html.HTMLParser(encoding=doc.original_encoding)
    doc = html.document_fromstring(content, parser=parser)
else:
    doc = html.parse(args.file)

pages = doc.xpath('//*[@class="ocr_page"]')
for page in pages:
    iname = args.imagesrc
    if args.basename:
        iname = os.path.join(args.basename, os.path.basename(iname))
    if not os.path.exists(iname):
        print("not found:", iname)
        sys.exit(1)
    image = Image.open(iname)
    lines = page.xpath("//*[@class='%s']" % args.element)
    lcount = 1
    for line in lines:
        bbox = [int(x) for x in get_prop(line, 'bbox').split()]
        if padding is not None:
            w, h = image.size
            bbox[0] = max(bbox[0]-padding[0], 0)
            bbox[1] = max(bbox[1]-padding[1], 0)
            bbox[2] = min(bbox[2]+padding[2], w)
            bbox[3] = min(bbox[3]+padding[3], h)
        if bbox[0] > bbox[2] or bbox[1] >= bbox[3]:
            continue
        lineimage = image.crop(bbox)
        out_img = args.pattern % lcount
        out_img_dir, out_img_name = os.path.split(out_img)
        if args.subfolder:
            out_img_dir = os.path.join(out_img_dir,args.subfolder)
            if not os.path.isdir(out_img_dir):
                os.mkdir(out_img_dir)
            out_img_name = out_img_name.replace("doc","cropped")
            out_img = os.path.join(out_img_dir,out_img_name)
        lineimage.save(out_img)
        lcount += 1
