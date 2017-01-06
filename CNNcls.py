import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from time import gmtime, strftime
import math
import os
from os import listdir
from os.path import isfile, join
import random
import PIL
from PIL import Image
from scipy import ndimage
from tensorflow.contrib.session_bundle import exporter
#import Group_B_Functions as bf
import sys
import Manual_Model_Creation
import pickle
import sys
import select
import tty
import termios
import time
from time import gmtime
from time import strftime
import datetime
from datetime import timedelta
FMT = '%H:%M:%S'

class CNNcls(object):

    def __init__(self,run_params=None):
        self.model_settings_for_saving = run_params
        if run_params == None:
            #### Initializes a basic model using settings from model_builder.py
            self.directory =  Manual_Model_Creation.get_build_directory()
            self.model_settings = Manual_Model_Creation.add_manually()
            run_parameters = self.model_settings

        else:
            self.directory = run_params['Build_Directory']
            run_parameters = run_params


        ############            Hard Coded Vars             ##########
        self.test_batch_size = 256  # don't worry about this
        self.input_num_channels = 1  # initial input images are greyscale
        self.num_classes = 2  # Only have 2 classifications, good and bad
        self.text_record = []  # Benchmarking List
        if 'dropout' in run_parameters.keys():
            self.dropout = run_parameters['dropout']
        else:
            self.dropout = 'f'

        ##### INITIZIALIZE MODEL PARAMETERS ####

        # Directory Stuff
        self.box_size = int(run_parameters['box_size'])

        self.build_dir = run_parameters['Build_Directory']
        self.image_path = run_parameters['Image_Dataset_Path']
        self.good_path = self.image_path + '/good' + '/res_' + str(self.box_size)
        self.bad_path = self.image_path + '/bad' + '/res_' + str(self.box_size)

        self.model_export_path = self.build_dir + '/trained_models'
        if not os.path.exists(self.model_export_path):
            print("Creating Model Export Directory")
            os.makedirs(self.model_export_path)
        self.save_model_path = self.model_export_path + '/CNN_T_' + run_parameters['name']

        # Global Paremeters
        self.batch_size = int(run_parameters['batch_size'])
        self.test_size = int(run_parameters['test_size'])
        self.num_iters = int(run_parameters['num_iters'])

        # Layer Config

        # Data Structure:
        # Layers is a dictionary stored in model params under key 'layers'
        # Two keys in layers, 'fcl_layers' and 'con_layers' which point to lists
        # Above each Contains a list of dictionaries, each element xxx_layers is a single layers configuration,
        # in a dictionary format

        layers = run_parameters['layers']
        self.con_layer_settings = layers['conv_layers']
        self.fcl_layer_settings = layers['fcl_layers']




        ### Test Directories are working
        self.test_filepaths(self.good_path,self.bad_path,self.box_size)



        ##################                 CREATE TENSORFLOW VARS              ######################
        print("Creating Tensorflow Vars")
        
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.input_num_channels], name='x_image')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')


        con_layer_settings = self.con_layer_settings
        fcl_layer_settings = self.fcl_layer_settings

        
        
        
        
        #### Create A dynamic Amount of conv layers as defined in Model Builder
        i = 0
        self.conv_layers = []
        num_conv_layers = len(con_layer_settings)
        first_layer_setting = con_layer_settings[i]
        print("Creating " + str(num_conv_layers) + " Convolutional Layers")
        temp_tensor = tf.placeholder(tf.float32, name='conv_layer' + str(i))
        
        temp_tensor = self.create_conv_layer(self.x_image, self.input_num_channels, first_layer_setting['filter_dim'],
                      first_layer_setting['num_filters'], 
                      use_max_pooling=True)
        
        first_con_layer = temp_tensor
        self.conv_layers.append(first_con_layer)
        ## Creating Further Layers
        i = 1
        while i < num_conv_layers:
            pre_layer_settings = con_layer_settings[i - 1]
            layer_settings = con_layer_settings[i]
            temp_tensor = tf.placeholder(tf.float32, name='conv_layer' + str(i))
            print(i)
            temp_tensor = self.create_conv_layer(self.conv_layers[i-1][0], pre_layer_settings['num_filters'],
                          layer_settings['filter_dim'],layer_settings['num_filters'],
                          use_max_pooling=True)
            
            self.conv_layers.append(temp_tensor)
            i += 1


        ##### Multistage or not multistage checker
        
        if run_parameters['MS'] == 'True':
            
            #### Create a flattening Layer from the first and last layers
            print("Creating multi-stage flattened layer")
            print("We connect the first conv layer and the last to  the fully connected layer")
            f1, f1n = self.flatten_layer(self.conv_layers[i - 1][0])
            f2, f2n = self.flatten_layer(self.conv_layers[0][0])
            values = [tf.pack(f1), tf.pack(f2)]
            pre_flat = tf.concat(1, values)

            # pre_flat = tf.stack(values, axis=0, name='stack')
            num_features = f2n + f1n
            self.layer_flat, self.num_features = self.flatten_layer(pre_flat)
            
        else:
            
            print("not using multistage")
            self.layer_flat, self.num_features = self.flatten_layer(self.conv_layers[i - 1][0])


        #### Create A dynamic Amount of FCL layers as defined in Model Builder
        
        i = 0
        self.fcl_layers = []
        num_fcl_layers = len(fcl_layer_settings)
        print('Creating ' + str(num_fcl_layers) + ' FCLs')
        first_layer_setting = fcl_layer_settings[i]
        temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i))
        
        temp_tensor = self.new_fc_layer(self.layer_flat, self.num_features, first_layer_setting['full_con_size'], use_relu=True)
   
        self.fcl_layers.append(temp_tensor)
        
        
        print("Created FCL connected to Flat Layer")
        # Create Middle ones
        
        i += 1
        while i < num_fcl_layers:
            print(i)
            print("Created intermediate layer")
            pre_layer_settings = fcl_layer_settings[i - 1]
            layer_settings = fcl_layer_settings[i]
            temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i))
            
            temp_tensor = self.new_fc_layer(self.fcl_layers[i - 1], pre_layer_settings['full_con_size'],
                          layer_settings['full_con_size'], use_relu=True)
            
            self.fcl_layers.append(temp_tensor)
            i += 1


        
        print("Created Final Layer Connects to softmax")
        # Below code connects to layer_fc3 twice so make sure to change both!        
        if self.dropout == 'True':
            self.keep_prob = tf.placeholder(tf.float32)
            self.dropout_layer = tf.nn.dropout(self.final_fcl_layer, self.keep_prob)
            W_fc2 = self.weight_variable([last_layer_setting['full_con_size'],self.num_classes])
            b_fc2 = self.bias_variable([self.num_classes])
            self.y_conv = tf.matmul(self.dropout_layer, W_fc2) + b_fc2
            
            
            self.y_pred = tf.nn.softmax(self.y_conv)

        else:
            # Created final  layer --> goes to softmax
            last_layer_setting = fcl_layer_settings[len(fcl_layer_settings)-1]
            temp_tensor = tf.placeholder(tf.float32, name='fcl_layer' + str(i-1))
            temp_tensor = self.new_fc_layer(self.fcl_layers[i-1], last_layer_setting['full_con_size'], 
                          self.num_classes, use_relu=False)
            self.fcl_layers.append(temp_tensor)
            self.final_fcl_layer = self.fcl_layers[len(self.fcl_layers) - 1]
            self.y_pred = tf.nn.softmax(self.final_fcl_layer)
            self.y_conv = self.final_fcl_layer
        
        
        
        
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y_true)  # for each image
            
        
        # combines all images to give single cost value which we aim to reduce, averages the result
        self.cost = tf.reduce_mean(self.cross_entropy) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.y_pred_class = tf.argmax(self.y_pred, dimension=1)
        self.y_true_class = tf.argmax(self.y_true, dimension=1)
        self.correct_prediction = tf.equal(self.y_pred_class, self.y_true_class)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        
        
        #End
        
        
    def return_files(self,directory):
        onlyfiles = [(directory + "/" + f) for f in listdir(directory) if isfile(join(directory, f))]
        return onlyfiles


    def train_model(self):
        print("Creating Testing Images")
        # these are the 10 testing samples we used (of size 500 testing images)
        self.files_training, self.files_testing = self.return_training_testing(self.good_path, self.bad_path)

        self.testing_images1, self.testing_labels1 = self.create_testing_batch(self.files_testing, self.test_size)
        self.testing_images2, self.testing_labels2 = self.create_testing_batch(self.files_testing, self.test_size)
        # testing_images3, testing_labels3 = create_testing_batch(files_testing, test_size)

        ########################################################################################################



        print("Init Session")
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        pickle.dump(self.model_settings_for_saving, open('list_of_runs.p', "wb"))

        
        
        
        #############THIS BEGINS THE TRAINING ################
        print("Beginnning Model Training")
        s1 = strftime("%H:%M:%S", gmtime())
        print(s1)
        self.run_batch()
        s2 = strftime("%H:%M:%S", gmtime())
        tdelta = datetime.datetime.strptime(s2, FMT) - datetime.datetime.strptime(s1, FMT)
        if tdelta.days < 0:
            tdelta = timedelta(days=0,
                               seconds=tdelta.seconds, microseconds=tdelta.microseconds)
        print(tdelta)

        ########### SAVE MODEL ###########
        save_path = saver.save(self.session, self.save_model_path + ".ckpt")
        print("Model saved in file: %s" % save_path)

        # RUN THE PROGRAM UP UNTIL HERE.
        # AFTER THE TRAINING IS COMPLETE YOU CAN RUN THE CODE BELOW
        # run these after the training has finished to check the accuracy. there are 10 samples to test
        acc = self.print_test_accuracy(self.testing_images1, self.testing_labels1)
        acc2 = self.print_test_accuracy(self.testing_images2, self.testing_labels2)
        # print_test_accuracy(testing_images3, testing_labels3)
        thelist = [acc,acc2,tdelta,self.model_settings_for_saving]
        file = pickle.dump(thelist, open(self.save_model_path + 'accuracy.p', "wb"))
        thefile = open(self.save_model_path+'stats.txt', 'w')
        for item in thelist:
            thefile.write("%s\n" % item)
            
        thefile.close        self.session.close()


    def run_batch(self):  # ensure you have defined global variables num_iters and files_training and batch_size
        nbc = NonBlockingConsole()
        for i in range(self.num_iters):
            if i < 20 or i %100 == 0:
                print(i)
                print("Creating batches")
                print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                
            batch_x, batch_y = self.create_training_batch(self.files_training, self.batch_size)
            if i in [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000]:
                print(i)

                # print_test_accuracy(testing_images1, testing_labels1)
            if self.dropout == 'True':
                 feed_dict_train = {self.x_image: batch_x, self.y_true: batch_y,self.keep_prob: 0.5}
            else:
                 feed_dict_train = {self.x_image: batch_x, self.y_true: batch_y}
            if i < 20 or i %100 == 0:
                print(i)
                print("running training run")
            if nbc.get_data() == '/x1b':
                print("stopping model training early")
                break
            
            self.session.run(self.optimizer, feed_dict=feed_dict_train)


    def test_filepaths(self,good_path,bad_path,box_size):
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
             self.image_size = B_size
             self.img_shape = (self.image_size, self.image_size)
         else:
             print("SIZE MISMATCH, CHECK DIMENSIONS OF TRAINING IMAGES AND MODEL PARAMETERS")
             exit(1)

    def plot_images(self,images, cls_true, cls_pred=None):  ###label of 0 is GOOD particle
        assert len(images) == len(cls_true) == 9
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


    def plot_image(self,image, cmap='plasma'):
        assert len(np.shape(image)) == 2
        _, ax1 = plt.subplots(1, 1)
        ax1.imshow(image, cmap=cmap)
        plt.show()


    def return_training_testing(self,good_dir, bad_dir, ratio=0.9):
        file_list_good = self.return_files(good_dir)  # list of all files (images)
        file_list_bad = self.return_files(bad_dir)
        for i, j in enumerate(file_list_good):
            file_list_good[i] = [j, np.array([1, 0])]
        for i, j in enumerate(file_list_bad):
            file_list_bad[i] = [j, np.array([0, 1])]
        file_list = file_list_good + file_list_bad
        random.shuffle(file_list)
        file_training = file_list[0:math.floor(ratio * len(file_list))]
        file_testing = file_list[math.floor(ratio * len(file_list)):]
        return file_training, file_testing  # lists containing ALL the training and testing files we have available


    def create_training_batch(self,file_training, batch_size):  # takes a list of all training files, and the batch size number
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
        image_array = image_array.reshape(batch_size, self.image_size, self.image_size, 1)
        label_array = label_array.reshape(batch_size, 2)
        return image_array, label_array


    def create_testing_batch(self,file_testing, test_size):
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
        image_array = image_array.reshape(test_size, self.image_size, self.image_size, 1)
        label_array = label_array.reshape(test_size, 2)
        return image_array, label_array


    def new_rand_weights(self,dims):
        return tf.Variable(tf.truncated_normal(dims, stddev=0.05))


    def new_rand_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))


    def create_conv_layer(self,input_tensor, num_input_channels, filter_size, num_filters,
                          use_max_pooling=True):  # 4-dim tensor: (image num, y-axis, x-axis, channel)

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.new_rand_weights(shape)
        biases = self.new_rand_biases(num_filters)

        layer = tf.nn.conv2d(input=input_tensor, filter=weights, strides=[1, 1, 1, 1],
                             padding='SAME')  # padding means width of output is same as width of input (by padding with zeros at edges), "same" padding
        layer += biases  # adds bias to each filter channel (so for each pixel of input we get convolution of filter (per channel) with pixel + bias)
        if use_max_pooling:
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # this halves the dimensions of the output
        layer = tf.nn.relu(layer)
        return layer, weights  # layer is a 4-d tensor, will be different from input as it will contain more channels now (dependent on the num of filters used)


    def flatten_layer(self,conv_layer):
        dims = conv_layer.get_shape()[1:4].num_elements()
        flat_conv_layer = tf.reshape(conv_layer, [-1, dims])
        return flat_conv_layer, dims


    def new_fc_layer(self,input, num_inputs, num_outputs,
                     use_relu=True):  # input is a 2-d tensor [num_images, num_inputs]l, where num_inputs will be pixel dims*num channels from output of last layer
        weights = self.new_rand_weights(dims=[num_inputs, num_outputs])
        biases = self.new_rand_biases(
            length=num_outputs)  # in this example it was 128 outputs, each of the weights in the weight matrix for some col is summed together and added with a bias to give the output of that particle output node (there are 128)
        layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)
        return layer






    def convert_value_to_goodbad(self,pred_list):
        new_list = []
        for i in pred_list:
            if i == 0:
                new_list.append("G")
            else:
                new_list.append("B")
        return new_list


    def plot_example_errors(self,cls_pred, correct, test_images, test_labels_class):
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

        cls_pred = self.convert_value_to_goodbad(cls_pred)
        cls_true = self.convert_value_to_goodbad(cls_true)

        # Plot the first 9 images.
        self.plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])


    def print_test_accuracy(self,test_images, test_labels):
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
            j = min(i + self.test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = test_images[i:j][:][:][:]

            # Get the associated labels.
            labels = test_labels[i:j][:]

            # Create a feed-dict with these images and labels.
            if self.dropout == 'True':
                feed_dict = {self.x_image: images,
                             self.y_true: labels,self.keep_prob: 1}            
            else:
                feed_dict = {self.x_image: images,
                             self.y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = self.session.run(self.y_pred_class, feed_dict=feed_dict)

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
        #print("Example errors:")
        #self.plot_example_errors(cls_pred, correct, test_images, testing_labels_class)
        return acc

    def plot_conv_weights(self,weights,session, input_channel=0):
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


    def plot_conv_layer(self,layer, image):
        # Assume layer is a TensorFlow op that outputs a 4-dim tensor
        # which is the output of a convolutional layer,
        # e.g. layer_conv1 or layer_conv2.

        # Create a feed-dict containing just one image.
        # Note that we don't need to feed y_true because it is
        # not used in this calculation.
        feed_dict = {self.x_image: [image]}

        # Calculate and retrieve the output values of the layer
        # when inputting that image.
        values = self.session.run(layer, feed_dict=feed_dict)

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



    def restore(self,path):
        self.new_saver = tf.train.import_meta_graph(path+".meta")
        self.new_saver.restore(self.session, path)
        print("Model restored.")
        self.all_vars = tf.trainable_variables()
    def load_testers(self):
        self.testing_images1, self.testing_labels1 = self.create_testing_batch(self.files_testing, self.test_size)
    def load_trainers(self):
        self.files_training, self.files_testing = self.return_training_testing(self.good_path, self.bad_path)

    def predict(self,imgar):
        #Takes numpy array of images, formatted as described by create testing batch and returns predicted values based on them

        test_images = imgar[0]
        labels = imgar[1]
        num_test = np.shape(test_images)[0]
        images = test_images
        feed_dict = {self.x_image: images,
                     self.y_true: labels}
        print("predicting...")
        cls_pred = self.session.run(self.y_pred_class, feed_dict=feed_dict)
        return cls_pred

    def predict_probs(self,imgar):
        #Takes numpy array of images, formatted as described by create testing batch and returns predicted values based on them

        test_images = imgar[0]
        labels = imgar[1]
        num_test = np.shape(test_images)[0]
        images = test_images
        feed_dict = {self.x_image: images,
                     self.y_true: labels}
        print("predicting...")
        cls_pred = self.session.run(self.y_pred, feed_dict=feed_dict)
        return cls_pred




    def create_predict_batch(self,files):
        #Takes list of files returns np array
        file_testing = files
        batch_list = []
        test_size = len(files)
        image_array = np.array([])
        label_array = np.array([])
        pixels =  max(np.array(Image.open(file_testing[0])).ravel())
        for i in file_testing:
            img = Image.open(i)
            im_array = np.array(img)
            im_array = im_array/pixels
            im_array = im_array.ravel()
            image_array = np.concatenate((image_array,im_array))
            label_array = np.concatenate((label_array,np.array([0, 0])))
        image_array = image_array.reshape(test_size, self.image_size, self.image_size, 1)
        label_array = label_array.reshape(test_size, 2)
        return image_array, label_array

    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def get_micrographs_png(self, file_testing, test_size):
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
        image_array = image_array.reshape(test_size, self.image_size, self.image_size, 1)
        label_array = label_array.reshape(test_size, 2)
        return image_array, label_array



class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False
