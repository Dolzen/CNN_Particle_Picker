import os
print(os.path.dirname(os.path.realpath(__file__)))



### DEAD CODE FOR FCL's AND UM CONVS


    #conv_layer1 = tf.placeholder(tf.float32, name="conv_layer1")
    #conv_layer2 = tf.placeholder(tf.float32, name="conv_layer2")
    #conv_layer3 = tf.placeholder(tf.float32, name="conv_layer3")
    #conv_layer4 = tf.placeholder(tf.float32, name="conv_layer4")

    #conv_layer1, conv_weights1 =

    #conv_layer2, conv_weights2 = create_conv_layer(conv_layer1, num_filters1, filter_dim2, num_filters2,
    #                                               use_max_pooling=True)

    #conv_layer3, conv_weights3 = create_conv_layer(conv_layer2, num_filters2, filter_dim3, num_filters3,
    #                                               use_max_pooling=True)

    #conv_layer4, conv_weights4 = create_conv_layer(conv_layer3, num_filters3, filter_dim4, num_filters4,
    #                                               use_max_pooling=True)

    #layer_flat = tf.placeholder(tf.float32, name="layer_flat")
    #layer_fc1 = tf.placeholder(tf.float32, name="layer_fc1")
    #layer_fc2 = tf.placeholder(tf.float32, name="layer_fc2")
    #layer_fc3 = tf.placeholder(tf.float32, name="layer_fc3")

    #layer_fc1 = new_fc_layer(layer_flat, num_features, full_con_size1, use_relu=True)
    #layer_fc2 = new_fc_layer(layer_fc1, full_con_size1, full_con_size2, use_relu=True)
    #layer_fc3 = new_fc_layer(layer_fc2, full_con_size2, num_classes, use_relu=False)


