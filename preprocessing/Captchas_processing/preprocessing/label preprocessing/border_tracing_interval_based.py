# --------------------------- A ADAPTER ---------------------------
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob
import cv2 as cv
import operator

# border tracing + interval-based
def calculate_corner_seg(thresh, nrootdir="C:/Users/MC/Desktop/PFE S5/Code/data/cut_image/"):
    #show_img = cv.imread('temp.jpg')
    #print("Thresh Shape:", thresh.shape, "Type:", thresh.dtype)
    contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    new_contours = []
    cur_contours = []
    filter_containor = []
    for i in contours:
        x, y, w, h = cv.boundingRect(i)   
        cur_contours.append([x, y, w, h])
    contours = sorted(cur_contours, key=operator.itemgetter(0))
    for i in range(0,len(contours)):  
        x = contours[i][0]
        y = contours[i][1]
        w = contours[i][2] 
        h = contours[i][3]
        newimage=thresh[y:y+h,x:x+w]
        nrootdir=("C:/Users/MC/Desktop/PFE S5/Code/data/cut_image/")
        if (h*w<80):
            continue
        new_contours.append(contours[i])
    t0 = [i[0] for i in new_contours]
    t1 = [i[1] for i in new_contours]
    t2 = [i[2] for i in new_contours]
    t3 = [i[3] for i in new_contours]
    x_max = max([new_contours[i][0] + new_contours[i][2] for i in range(len(new_contours))] )+5
    x_min = min(t0)-5
    y_max = 59
    y_min = 0
    if(x_min<0):
        x_min = 0
    if(x_max>159):
        x_max = 159
    width = (x_max-x_min)//4
    for i in range(0,4):
        newimage=thresh[y_min:y_max,x_min+i*width:x_min+(i+1)*width]
        top, bottom, left, right = [1]*4
        newimage = cv.copyMakeBorder(newimage, top, bottom, left, right, cv.BORDER_CONSTANT)
        # newimage = cv.resize(newimage,(30, 60), interpolation = cv.INTER_CUBIC)
        cv.imwrite( "temp.jpg",newimage)
        filter_containor.append(Image.open("temp.jpg"))
    return filter_containor


# Load the colored image
image = cv.imread("C:/Users/MC/Desktop/PFE S5/Code/data/banknote_unsharp_mask/0bb281b845f0eb07c8c42289208ef5d2_text_image.png", cv.IMREAD_GRAYSCALE)

print("Image Shape:", image.shape, "Type:", image.dtype)  # Expected: (H, W), uint8

# Pass to function
processed_images = calculate_corner_seg(image)

# Display results
for img in processed_images:
    img.show()  # Opens each processed character as a PIL image