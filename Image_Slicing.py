#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:22:12 2019

@author: ErichBrandstetter
"""

import image_slicer
from PIL import Image
import os

inpath = "/Volumes/Extern/Bulk_Images/NAIP/unzipped_images"
os.chdir(inpath)

outpath = "/Volumes/Extern/Bulk_Images/NAIP/reduced_images"

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

for i,img in enumerate(mylistdir(inpath)):
    input_path = os.path.join(inpath, img)
    im = Image.open(input_path)
    im.convert("RGB")
    os.chdir(outpath)
    im.save("image{}.jpeg".format(i), "JPEG", dpi = (60,60))

newpath = "/Volumes/Extern/Bulk_Images/NAIP/sliced_images"

for i,img in enumerate(mylistdir(outpath)):
    new_input = os.path.join(outpath, img)
    tiles = image_slicer.slice(new_input, 40)
    image_slicer.save_tiles(tiles, directory = newpath,
                            prefix = 'slice{}'.format(i), format = 'JPEG')
                            
