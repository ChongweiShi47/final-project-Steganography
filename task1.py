# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:15:53 2021

@author: 13732
"""
from PIL import Image
from steganography import Steganography
merged_image=Steganography.merge(Image.sopen('img1.jpg'),Image.open('img2.jpg'))
img1=Image.open('img1.jpg')
img2=Image.open('img2.jpg')
merged_image=Steganography.merge(img1,img2)

merged_image
unmerged=Steganography.unmerge(merged_image)
unmerged
img2