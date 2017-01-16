import Group_B_Functions as bf
import CNNcls
import tensorflow as tf
import Group_B_Functions as bf
import PIL
from PIL import Image
import numpy as np
import random
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
import shutil
import os
import sys
def tileslice(x,y,width,micrograph):
    #Creates a tile of size width*width from the given micrograph numpy array
    i_min = x - (width//2)
    i_max = x + (width//2)
    j_min = y - (width//2)
    j_max = y + (width//2)
    M_selection = np.array(micrograph[i_min:i_max, j_min:j_max])
    return M_selection

def main():
    # Sliding Window Tile picker, this script performs ops at a LOWER RESOLUTION, only use this
    # Script if you want to pick at a resolution lower than the original
    # The images returned will be at full resolution, but the actual picking is done on scaled
    # down versions of the micrographs

    base_res = 240
    scaled_res = 60
    tile_dim = 240  # The resolution of the original image
    tile_size = 100  # Change to adjust size of area predicted at once
    num_tiles = 100  # number of tiles we're going to pick
    threshold = 0.8  # Probability threshold, if cls_pred > threshold we will pick that tile as a good candidate
    
    model_name = 'CNN_T_test1'


    base_model_path = dir_path + '/trained_models/' + model_name
    scaled_model_path = base_model_path
    base_micrograph_path = dir_path + '/Images/micrographs'


    print("Sliding Window Tile Particle Picker, by Ed")

    scl_path = base_micrograph_path + '/res_' + str(scaled_res)
    original_images = files = bf.return_files(base_micrograph_path)
    scaled_images = bf.return_files(scl_path)
    # Needs to be fixed ASAP when model is done training
    ###################################################
    # Initiliaze model
    model = CNNcls.CNNcls()
    path = base_model_path + '.ckpt'
    model.restore(path)
    ###################################################


    # Get folder of micrographs
    # First get scaled micrograph
    if os.path.isdir(scl_path) == False:  # Check if folder of scaled micrographs exists
        print('Scaled down folder of micrographs not found, would you like to create one?')
        # raw_input returns the empty string for "enter"
        yes = set(['yes', 'y', 'ye', ''])
        no = set(['no', 'n'])

        choice = input().lower()
        if choice in yes:
            # Create all images
            os.makedirs(scl_path)
            # iteratre through all images
            count = 0
            for image in original_images:
                img = Image.open(image)
                path = scl_path + '/' + os.path.basename(image)[:-4] + '_res_' + str(scaled_res) + '.png'
                print(path)
                if count % 100 == 0:
                    print(path)
                img.save(path, dpi=(scaled_res, scaled_res))
                count += 1
        elif choice in no:
            print('well no way to proceed, quitting')
            exit()
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")
    if len(scaled_images) != len(original_images):
        print('Not all of the scaled micrograph images were found in the folder, recreate scaled folder?')
        choice = input().lower()
        if choice in yes:
            shutil.rmtree(scl_path)
            # Create all images
            os.makedirs(scl_path)
            # iteratre through all images
            count = 0
            for image in original_images:
                img = Image.open(image)
                path = scl_path + '/' + os.path.basename(image)[:-4] + '_res_' + str(scaled_res) + '.png'
                print(path)
                if count % 100 == 0:
                    print(path)
                img.save(path, dpi=(scaled_res, scaled_res))
                count += 1
        elif choice in no:
            print('proceeding with what was found in the scaled path')
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")

    # change for iteration over folder
    ########################################################################
    files = scaled_images
    mpath = files[0]
    scaled_micrograph = Image.open(mpath)  # Open a single Micrograph for testing
    orig_micrograph = original_images[0]
    ################################################################################################


    # Set up Parameters
    test_size = tile_size * tile_size
    batch_size = tile_size * tile_size
    # Turn the micrograph into a numpy array
    mgar = np.array(scaled_micrograph)
    xres, yres = mgar.shape
    # Create Probability Matrix
    pMatrix = np.zeros((xres, yres))
    coords_list = create_rand_list(scaled_res,scaled_micrograph)
    old_img = None
    while len(coords_list) > 1:
        coords_list,current_batch = create_rand_list(scaled_res,batch_size,scaled_micrograph)
        cls_pred = model.predict_probs(imgar)  # Run model and insert probability of positive into pMatrix
        old_img = predict_and_reveal(current_batch,scaled_micrograph,cls_pred,old_img=old_img)
    
def create_rand_list(size,micrograph):
    shape = micrograph.shape
    x_list = random(size//2,shape[0] - size//2)
    y_list = random(size//2,shape[1] - size//2)
    i = 0
    rands = []
    while i < len(x_list):
        rands.append(x_list[i],y_list[i])
    return rands

def create_batch(coords_left,batch_size,micrograph,box_size):
    #Creates a batch ready to be given to feed dict
    
    pixels = max(micrograph.ravel())
    if batch_size > len(coords_left):
        batch_size = len(coords_left)
    current_coords = coords_left[:batch_size]
    coords_left = coords_left[batch_size:]
    i = 0
    image_list = []
    while i < len(current_coords):
        x = current_coords[i][0]
        y = current_coords[i][1]
        image_list.append(x,y,tileslice(x,y,box_size,micrograph))
        i += 1
    a = 0
    label_array = np.array([])
    pre_batch = np.array([])
    while a < batch_size:
        im_array = image_list[a][2]
        im_array = im_array / pixels
        im_array = im_array.ravel()
        pre_batch = np.concatenate((pre_batch, im_array))
        label_array = np.concatenate((label_array, np.array([0, 0])))
        a += 1
    label_array = label_array.reshape(batch_size, 2)
    current_batch = pre_batch.reshape(batch_size, box_size, box_size, 1)
    imgar = current_batch, label_array
    return coords_left,imgar


    return coords_left,image_list


def random_sample(micrograph,num,size):
    #Samples random co-ords of a micrograph returns a list of [(image,x,y),(image,x,y)]
    length = micrograph.shape[0]
    width = micrograph.shape[1]
    x_list = random(size/2, length-size//2,num)
    y_list = random(size/2, width-size//2, num)
    i = 0
    images = []
    while i < num:
        current_element = (x_list[i],y_list[i])
        current_element.append(tilestack(x_list[i],y_list[i],size,micrograph))
        images.append(current_element)
        i += 1
    return images

def predict_and_reveal(images,micrograph,cls_pred,old_img=None):
    #Colors a copy of the micrograph based on results of CNN
    i = 0
    for items in cls_pred:
        images[i].append(items)
        i += 1
    if old_img == None:
        revealed = np.zeros(micrograph.shape)
    else:
        revealed = old_img
    for items in images:
        x = items[0]
        y = items[1]
        i_min = x - (scaled_res // 2)
        i_max = x + (scaled_res // 2)
        j_min = y - (scaled_res // 2)
        j_max = y + (scaled_res // 2)
        if items[3] > 0.8:
            revealed[i_min:i_max, j_min:j_max] = items[2]
    return(revealed)
    revealed.show
