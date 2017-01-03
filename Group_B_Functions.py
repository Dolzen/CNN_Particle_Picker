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

export_path = sys.argv[-1]
start = time.time()
end = time.time()
print(end - start)

good_path = '/home/serv/Downloads/Images/good'  ####CHANGE THIS TO YOUR DIRECTORY FOR THE IMAGES!!!####################
bad_path = '/home/serv/Downloads/Images/BAD240FULL'  ####CHANGE THIS TO YOUR DIRECTORY FOR THE IMAGES!!!####################
image_size = 240
img_shape = (image_size, image_size)


##################FUNCTION PROTOTYPING####################


def return_files(directory):
    onlyfiles = [(directory + "/" + f) for f in listdir(directory) if isfile(join(directory, f))]
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


def create_training_batch(file_training, batch_size):  # takes a list of all training files, and the batch size number
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


def get_micrographs_png(file_testing, test_size):
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
        feed_dict = {x_image: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_class, feed_dict=feed_dict)

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


def plot_conv_weights(weights,session, input_channel=0):
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
