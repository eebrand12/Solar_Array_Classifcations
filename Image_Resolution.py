#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:10:32 2019

@author: ErichBrandstetter
"""

from PIL import Image
import os

#print(os.getcwd())
path = "/Users/ErichBrandstetter/Documents/DA_Satellite_Images/High_Res_Images/"
os.chdir(path)
#print (os.getcwd())

#im = Image.open("m_3008805_ne_16_060_20181204.tif") #open image

new_path = "/Users/ErichBrandstetter/Documents/DA_Satellite_Images/Low_Res_Images/"

def conversion(im):
    """converts images from high resolution to low resolution and saves
    in new folder"""
    im.convert("RGB")
    os.chdir(new_path)# Set outfile path
    im.save("test.jpeg","JPEG", dpi = (600,600)) #Save off low Res Image
    
for img in os.listdir("./"): #loop through high res images
    os.chdir(path)
    im = Image.open(img) #open images
    conversion(im) #convert image

#test_image = Image.open("{}test.jpeg","r").format(new_path)

