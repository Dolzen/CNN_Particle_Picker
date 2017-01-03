import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import math
import os
from os import listdir
from os.path import isfile, join
import random
import PIL
from PIL import Image
from scipy import ndimage
from tensorflow.contrib.session_bundle import exporter
import sys
import Group_B_Functions as bf



# def convert_images_res(good_path,bad_path,res)
res = 120
ores = 240
image_size = 240
sres = str(res)
#This creates a subfolder and copies all images in goodpath and badpath into 
#Said folder at a lower resolution
#Paths to get images from
#Relative path to create folder and place converted images into
good_path = '/home/serv/Build_1/Images/good'  ####CHANGE THIS TO YOUR DIRECTORY FOR THE IMAGES!!!####################
bad_path = '/home/serv/Build_1/Images/bad'  ####CHANGE THIS TO YOUR DIRECTORY FOR THE IMAGES!!!####################


print(image_size)
img_shape = (image_size, image_size)
newgood = good_path + '/res_' + sres +'/'
newbad =  bad_path  + '/res_' + sres + '/'
goodfiles = bf.return_files(good_path + '/res_' + str(ores))    
badfiles = bf.return_files(bad_path + '/res_' + str(ores))     

#Create new images
if not os.path.exists(newgood):
    print('making dir')
    os.makedirs(newgood)
    #iteratre through all images
    count = 0
    for image in goodfiles:
        img = Image.open(image)
        path = newgood + os.path.basename(image)[:-4] + '_res_' + sres +'.png'
        if count%100 == 0:
            print(path)
        img = img.resize((res,res),Image.ANTIALIAS)
        img.save(path, dpi=(res,res))
        count += 1
    
if not os.path.exists(newbad):
    print('making dir')
    os.makedirs(newbad)

    for image in badfiles:
        img = Image.open(image)
        path = newbad + os.path.basename(image)[:-4] + '_res_' + sres +'.png'
        if count%100 == 0:
            print(path)
        img = img.resize((res,res),Image.ANTIALIAS)
        img.save(path, dpi=(res,res))

    #Return newgood,newbad
