import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import math
from os import listdir
from os.path import isfile, join
import random
import PIL
from PIL import Image
from scipy import ndimage
from tensorflow.contrib.session_bundle import exporter
import sys
import os

 
def run_model(run_parameters):
    ############            Hard Coded Vars             ##########
    test_batch_size = 256  # don't worry about this
    input_num_channels = 1  # initial input images are greyscale
    num_classes = 2  # Only have 2 classifications, good and bad
    text_record = []  # Benchmarking List
    
    
    
    
    ##### INITIZIALIZE MODEL PARAMETERS ####

    # Directory Stuff
    box_size = int(run_parameters['box_size'])
    
    
    build_dir = run_parameters['Build_Directory']
    image_path = run_parameters['Image_Dataset_Path']
    good_path = image_path +'/good' + '/res_' + str(box_size)
    bad_path = image_path +'/bad'  + '/res_' + str(box_size)

    model_export_path = build_dir + '/trained_models'
    if not os.path.exists(model_export_path):
        print("Creating Model Export Directory")
        os.makedirs(model_export_path)
    save_model_path = model_export_path +'/CNN_T_'+ run_parameters['name']


    #Global Paremeters
    batch_size = int(run_parameters['batch_size'])
    test_size = int(run_parameters['test_size'])
    num_iters = int(run_parameters['num_iters'])



    # Layer Config

    #Data Structure:
    # Layers is a dictionary stored in model params under key 'layers'
    # Two keys in layers, 'fcl_layers' and 'con_layers' which point to lists
    # Above each Contains a list of dictionaries, each element xxx_layers is a single layers configuration,
    # in a dictionary format

    layers = run_parameters['layers']
    con_layer_settings = layers['conv_layers']
    fcl_layer_settings = layers['fcl_layers']







    
    
    
    
    #### Test Image Paths are working

    Atest_path = os.listdir(bad_path)[1]
    Btest_path = os.listdir(good_path)[1]

    print('Testing Bad File Path')
    print(good_path)
    testA = Image.open(bad_path + '/' + Atest_path)
    testA = np.array(testA)
    A_size = testA.shape[0]

    print('Testing Good File Path')
    print(good_path)
    testB = Image.open(good_path + '/' + Btest_path)
    testB = np.array(testB)
    B_size = testB.shape[0]

    if B_size == A_size and B_size == int(box_size):
        print('Image sizes match with box size')
        image_size = B_size
        img_shape = (image_size, image_size)
    else:
        print("SIZE MISMATCH, CHECK DIMENSIONS OF TRAINING IMAGES AND MODEL PARAMETERS")
        exit(1)

    ##################FUNCTION PROTOTYPING####################


    def return_files(directory):
        onlyfiles = [(directory + "/" + f) for f in listdir(directory) if isfile(join(directory, f))]
        # print('returning files')
        # print(onlyfiles)
        return onlyfiles

    def plot_images(images, cls_true, cls_pred=None):  ###label of 0 is GOOD particle
        assert len(images) == len(cls_true) == 9
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(img_shape), cmap='binary')
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def plot_image(image, cmap='plasma'):
        assert len(np.shape(image)) == 2
        _, ax1 = plt.subplots(1, 1)
        ax1.imshow(image, cmap=cmap)
        plt.show()

    def return_training_testing(good_dir, bad_dir, ratio=0.9):
        file_list_good = return_files(good_dir)  # list of all files (images)
        file_list_bad = return_files(bad_dir)
        for i, j in enumerate(file_list_good):
            file_list_good[i] = [j, np.array([1, 0])]
        for i, j in enumerate(file_list_bad):
            file_list_bad[i] = [j, np.array([0, 1])]
        file_list = file_list_good + file_list_bad
        random.shuffle(file_list)
        file_training = file_list[0:math.floor(ratio * len(file_list))]
        file_testing = file_list[math.floor(ratio * len(file_list)):]
        return file_training, file_testing  # lists containing ALL the training and testing files we have available

    def create_training_batch(file_training,
                              batch_size):  # takes a list of all training files, and the batch size number
        batch_size = int(batch_size)
        batch_index = [random.randint(0, len(file_training) - 1) for r in range(batch_size)]
        batch_list = []
        for i in batch_index:
            batch_list.append(file_training[i])
        pixels = max(np.array(Image.open(file_training[0][0])).ravel())
        image_array = np.array([])
        label_array = np.array([])
        for i in batch_list:
            img = Image.open(i[0])
            im_array = np.array(img)
            im_array = im_array / pixels
            im_array = im_array.ravel()
            image_array = np.concatenate((image_array, im_array))
            label_array = np.concatenate((label_array, i[1]))
        image_array = image_array.reshape(batch_size, image_size, image_size, 1)
        label_array = label_array.reshape(batch_size, 2)
        return image_array, label_array

    def create_testing_batch(file_testing, test_size):
        batch_index = [random.randint(0, len(file_testing) - 1) for r in range(test_size)]
        batch_list = []
        for i in batch_index:
            batch_list.append(file_testing[i])
        pixels = max(np.array(Image.open(file_testing[0][0])).ravel())
        image_array = np.array([])
        label_array = np.array([])
        for i in batch_list:
            img = Image.open(i[0])
            im_array = np.array(img)
            im_array = im_array / pixels
            im_array = im_array.ravel()
            image_array = np.concatenate((image_array, im_array))
            label_array = np.concatenate((label_array, i[1]))
        image_array = image_array.reshape(test_size, image_size, image_size, 1)
        label_array = label_array.reshape(test_size, 2)
        return image_array, label_array

    def new_rand_weights(dims):
        return tf.Variable(tf.truncated_normal(dims, stddev=0.05))

    def new_rand_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def create_conv_layer(input_tensor, num_input_channels, filter_size, num_filters,
                          use_max_pooling=True):  # 4-dim tensor: (image num, y-axis, x-axis, channel)

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = new_rand_weights(shape)
        biases = new_rand_biases(num_filters)

        layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=[1, 1, 1, 1],
                             padding='SAME')  # padding means width of output is same as width of input (by padding with zeros at edges), "same" padding
        layer += biases  # adds bias to each filter channel (so for each pixel of input we get convolution of filter (per channel) with pixel + bias)
        if use_max_pooling:
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # this halves the dimensions of the output
        layer = tf.nn.relu(layer)
        return layer, weights  # layer is a 4-d tensor, will be different from input as it will contain more channels now (dependent on the num of filters used)

    def flatten_layer(conv_layer):
        dims = conv_layer.get_shape()[1:4].num_elements()
        flat_conv_layer = tf.reshape(conv_layer, [-1, dims])
        return flat_conv_layer, dims

    def new_fc_layer(input, num_inputs, num_outputs,
                     use_relu=True):  # input is a 2-d tensor [num_images, num_inputs]l, where num_inputs will be pixel dims*num channels from output of last layer
        weights = new_rand_weights(dims=[num_inputs, num_outputs])
        biases = new_rand_biases(
            length=num_outputs)  # in this example it was 128 outputs, each of the weights in the weight matrix for some col is summed together and added with a bias to give the output of that particle output node (there are 128)
        layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

    def convert_value_to_goodbad(pred_list):
        new_list = []
        for i in pred_list:
            if i == 0:
                new_list.append("G")
            else:
                new_list.append("B")
        return new_list

    def plot_example_errors(cls_pred, correct, test_images, test_labels_class):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = test_images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = test_labels_class[incorrect]

        cls_pred = convert_value_to_goodbad(cls_pred)
        cls_true = convert_value_to_goodbad(cls_true)

        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])

    def print_test_accuracy(test_images, test_labels):
        # Number of images in the test-set.
        num_test = np.shape(test_images)[0]

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = test_images[i:j][:][:][:]

            # Get the associated labels.
            labels = test_labels[i:j][:]

            # Create a feed-dict with these images and labels.

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        testing_labels_class = np.argmax(test_labels, axis=1)
        # Convenience variable for the true class-numbers of the test-set.
        cls_true = testing_labels_class

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))
        print("Example errors:")
        plot_example_errors(cls_pred, correct, test_images, testing_labels_class)

    def plot_conv_weights(weights, input_channel=0):
        # Assume weights are TensorFlow ops for 4-dim variables
        # e.g. weights_conv1 or weights_conv2.

        # Retrieve the values of the weight-variables from TensorFlow.
        # A feed-dict is not necessary because nothing is calculated.
        w = session.run(weights)

        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(w)
        w_max = np.max(w)

        # Number of filters used in the conv. layer.
        num_filters = w.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = w[:, :, input_channel, i]

                # Plot image.
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap='seismic')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_conv_layer(layer, image):
        # Assume layer is a TensorFlow op that outputs a 4-dim tensor
        # which is the output of a convolutional layer,
        # e.g. layer_conv1 or layer_conv2.

        # Create a feed-dict containing just one image.
        # Note that we don't need to feed y_true because it is
        # not used in this calculation.
        feed_dict = {x_image: [image]}

        # Calculate and retrieve the output values of the layer
        # when inputting that image.
        values = session.run(layer, feed_dict=feed_dict)

        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i < num_filters:
                # Get the output image of using the i'th filter.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, interpolation='nearest', cmap='binary')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


    ##########      BUILDING THE TENSORFLOW GRAPH       #######################


    ##############  CREATE TENSORFLOW VARS              #################################

    print("Creating Tensorflow Vars")
    x_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, input_num_channels], name='x_image')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')




    # NEW CODE



    #### Create A dynamic Amount of conv layers as defined in Model Builder
    
    conv_layers = []
    num_conv_layers = len(con_layer_settings)
    print("Creating " + str(num_conv_layers) + "Convolutional Layers")
    i = 0
    first_layer_setting = con_layer_settings[i]
    temp_tensor = tf.placeholder(tf.float32, name='conv_layer' + str(i))
    temp_tensor = create_conv_layer(x_image, input_num_channels, first_layer_setting['filter_dim'], first_layer_setting['num_filters'],
                                                   use_max_pooling=True)
    conv_layers.append(temp_tensor)

    i = 1
    while i < num_conv_layers:
        pre_layer_settings = con_layer_settings[i-1]
        layer_settings = con_layer_settings[i]
        temp_tensor = tf.placeholder(tf.float32,name='conv_layer' + str(i))
        temp_tensor = create_conv_layer(x_image, input_num_channels, pre_layer_settings['filter_dim'], pre_layer_settings['num_filters'],
                                                   use_max_pooling=True)
        conv_layers.append(temp_tensor)
        i += 1



    #### Create a flattening Layer
    layer_flat, num_features = flatten_layer(conv_layers[i-1][0])


    #### Create A dynamic Amount of FCL layers as defined in Model Builder
    
    
    #Created Connected to the flat layer
    
    fcl_layers = []
    num_fcl_layers = len(fcl_layer_settings)
    print('Creating ' + str(num_fcl_layers) + ' FCLs')
    i = 0
    first_layer_setting = fcl_layer_settings[i]
    temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i))

    temp_tensor = new_fc_layer(layer_flat, num_features, first_layer_setting['full_con_size'], use_relu=True)
    fcl_layers.append(temp_tensor)
    print("Created FCL connected to Flat Layer")

    # Create Middle ones
    i += 1
    while i < num_fcl_layers:
        print("Created intermediate layer")
        pre_layer_settings = fcl_layer_settings[i - 1]
        layer_settings = fcl_layer_settings[i]
        temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i))
        temp_tensor = new_fc_layer(fcl_layers[i-1], pre_layer_settings['full_con_size'], layer_settings['full_con_size'], use_relu=True)
        fcl_layers.append(temp_tensor)
        i += 1

        
    #Created final  layer --> goes to softmax
    i -= 1
    last_layer_setting = fcl_layer_settings[i]
    temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i))

    temp_tensor = new_fc_layer(fcl_layers[i], last_layer_setting['full_con_size'],num_classes, use_relu=False)
    fcl_layers.append(temp_tensor)
    print("Created Final Layer Connects to softmax")




    # New Code , making the number of layers modular! Yay!
    # Below code connects to layer_fc3 twice so make sure to change both!
    final_fcl_layer = fcl_layers[len(fcl_layers)-1]
    y_pred = tf.nn.softmax(final_fcl_layer)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_fcl_layer, labels=y_true)  # for each image
    cost = tf.reduce_mean(
        cross_entropy)  # combines all images to give single cost value which we aim to reduce, averages the result
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    # optimizer2 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    y_pred_class = tf.argmax(y_pred, dimension=1)
    y_true_class = tf.argmax(y_true, dimension=1)

    correct_prediction = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ########################################################################################################
    print("Creating Testing Images")
    # these are the 10 testing samples we used (of size 500 testing images)
    files_training, files_testing = return_training_testing(good_path, bad_path)

    testing_images1, testing_labels1 = create_testing_batch(files_testing, test_size)
    testing_images2, testing_labels2 = create_testing_batch(files_testing, test_size)
    #testing_images3, testing_labels3 = create_testing_batch(files_testing, test_size)
    #testing_images4, testing_labels4 = create_testing_batch(files_testing, test_size)
    #testing_images5, testing_labels5 = create_testing_batch(files_testing, test_size)
    print('halfway!')
    #testing_images6, testing_labels6 = create_testing_batch(files_testing, test_size)
    #testing_images7, testing_labels7 = create_testing_batch(files_testing, test_size)
    #testing_images8, testing_labels8 = create_testing_batch(files_testing, test_size)
    #testing_images9, testing_labels9 = create_testing_batch(files_testing, test_size)
    #testing_images10, testing_labels10 = create_testing_batch(files_testing, test_size)

    ########################################################################################################



    ######## CREATE TF SESSION AND INIT MODEL ##########
    print("Init Session")
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    # Initial val = 1000

    start_t_length = time.process_time()

    def run_batch():  # ensure you have defined global variables num_iters and files_training and batch_size
        for i in range(num_iters):
            batch_x, batch_y = create_training_batch(files_training, batch_size)
            if i in [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
                print(i)
                # print_test_accuracy(testing_images1, testing_labels1)
            feed_dict_train = {x_image: batch_x, y_true: batch_y}
            session.run(optimizer, feed_dict=feed_dict_train)

    # print(session.run(accuracy, feed_dict={x_image: testing_images1, y_true: testing_labels1}))           ###THIS SHOULD GIVE ABOUT 50% accuracy  WHEN RUN





    #############THIS BEGINS THE TRAINING ################
    print("Beginnning Model Training")
    run_batch()

    end_t_length = time.process_time()
    elps_time = end_t_length - start_t_length
    text_record.append(['Time taken for full training', elps_time])

    ########### SAVE MODEL ###########

    save_path = saver.save(session, save_model_path + ".ckpt")
    print("Model saved in file: %s" % save_path)

    # RUN THE PROGRAM UP UNTIL HERE.
    # AFTER THE TRAINING IS COMPLETE YOU CAN RUN THE CODE BELOW

    # run these after the training has finished to check the accuracy. there are 10 samples to test
    print_test_accuracy(testing_images1, testing_labels1)
    print_test_accuracy(testing_images2, testing_labels2)
    #print_test_accuracy(testing_images3, testing_labels3)
    #print_test_accuracy(testing_images4, testing_labels4)
    #print_test_accuracy(testing_images5, testing_labels5)
    #print_test_accuracy(testing_images6, testing_labels6)
    #print_test_accuracy(testing_images7, testing_labels7)
    #print_test_accuracy(testing_images8, testing_labels8)
    #print_test_accuracy(testing_images9, testing_labels9)
    #print_test_accuracy(testing_images10, testing_labels10)
    print(text_record)

    session.close()

