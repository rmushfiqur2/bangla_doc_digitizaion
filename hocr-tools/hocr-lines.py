#!/usr/bin/env python

# extract the text within all the ocr_line elements within the hOCR file

from __future__ import print_function
import argparse
import re
import sys

from lxml import html

parser = argparse.ArgumentParser(
    description=('extract the text within all the ocr_line elements '
                 'within the hOCR file')
)
parser.add_argument('file', nargs='?', default=sys.stdin)
args = parser.parse_args()

doc = html.parse(args.file)

for line in doc.xpath("//*[@class='ocr_line']"):
    print(re.sub(r'\s+', '\x20', line.text_content()).strip())
