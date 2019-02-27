# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:10:16 2018

@author: HHY
"""
import TensorflowUtils as utils
import tensorflow as tf
import numpy as np
#source image
size1=1
#annotation image
size=size1*4
FLAGS = tf.flags.FLAGS
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
tf.flags.DEFINE_string("model_dir", "../Model/16/", "Path to vgg model mat")
tf.flags.DEFINE_integer("NUM_OF_CLASSESS",3,"numbers of classes")
tf.flags.DEFINE_integer("IMAGE_SIZE",1024/size1,"length of image")
'''
#vgg pool5 [32,32,512]
InputFilters=512
#ResUnetFilters=[512,1024,2048,4096]
ResUnetFilters=[256,512,1024,2048]

#vgg pool4 [64,64,512]
InputFilters=512
ResUnetFilters=[256,512,1024,2048]

#vgg pool3 [128,128,256]
InputFilters=256
ResUnetFilters=[128,256,512,1024]

#vgg pool2 [256,256,128]
InputFilters=128
ResUnetFilters=[64,128,256,512]

#vgg pool1 [512,512,64]
InputFilters=64
ResUnetFilters=[32,64,128,256]
'''

#vgg pool2 [256,256,128]
InputFilters=128
ResUnetFilters=[64,128,256,512]

def vgg_net(weights, image,layers_num=16):
    #VGG16
    if layers_num==16:
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3','relu3_3', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3','relu4_3', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3','relu5_3'
        )
    #VGG19
    if layers_num==19:
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    return net,layers


def inference(image,final_layer, keep_prob,isDeconv=True):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
                : Image value range from 0 to 2(In our dataset 0 represents background ,1 represents abnormal,2 represents normal)
    :param final_layer:eg,'pool5'means using the first 5 layers of the pre-trained vgg model's weights
                      :[this values can be pool3,pool5]
    :param:isDeconv:use or not use the deconvolutional layers to up-sampling
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    print("pre-processed images ...")
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)

    image_net,layers = vgg_net(weights, processed_image)
    if final_layer=='pool5':
        conv_final_layer = image_net['conv5_3']
    else:
        pos=layers.index(final_layer)
        conv_final_layer = image_net[layers[pos-2]]
    featuremap = utils.max_pool_2x2(conv_final_layer)

    with tf.variable_scope("inference"):
#####################################Encoder Block 1 #####################################

        to_decoder = []
        Filters1=ResUnetFilters[0]
        Filters2=ResUnetFilters[1]
        Filters3=ResUnetFilters[2]
        Filters4=ResUnetFilters[3]
        #conv1
        W1 = utils.weight_variable([3,3,InputFilters,Filters1],name="W1")
        b1 = utils.bias_variable([Filters1],name="b1")
        conv1 = utils.conv2d_basic(featuremap, W1, b1)

        bn1=utils.batch_norm(conv1, n_out=Filters1,phase_train=tf.cast(True, tf.bool),scope="bn1")
        relu1 = tf.nn.relu(bn1, name="relu1")
        if FLAGS.debug:
            utils.add_activation_summary(relu1)
        #relu_dropout1= tf.nn.dropout(relu1, keep_prob=keep_prob)

        #conv2
        W2 = utils.weight_variable([3,3,Filters1,Filters1],name="W2")
        b2 = utils.bias_variable([Filters1],name="b2")
        conv2 = utils.conv2d_basic(relu1, W2, b2)

        #resnet addition
        W_Add1 = utils.weight_variable([1,1,InputFilters,Filters1],name="W_Add1")
        b_Add1 = utils.bias_variable([Filters1],name="b_Add1")
        conv_Add1=utils.conv2d_basic(featuremap, W_Add1, b_Add1)
        bn_Add1=utils.batch_norm(conv_Add1, n_out=Filters1,phase_train=tf.cast(True, tf.bool),scope="bn_Add1")
        Addition1=bn_Add1+conv2

        to_decoder.append(Addition1)
#####################################Encoder Block 2 #####################################

        bn2=utils.batch_norm(Addition1, n_out=Filters1,phase_train=tf.cast(True, tf.bool),scope="bn2")
        relu2 = tf.nn.relu(bn2, name="relu2")
        #relu_dropout2= tf.nn.dropout(relu2, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu2)

        W3 = utils.weight_variable([3,3,Filters1,Filters2],name="W3")
        b3 = utils.bias_variable([Filters2],name="b3")
        conv3 = utils.conv2d_strided(relu2, W3, b3)

        bn3=utils.batch_norm(conv3, n_out=Filters2,phase_train=tf.cast(True, tf.bool),scope="bn3")
        relu3 = tf.nn.relu(bn3, name="relu3")
        if FLAGS.debug:
            utils.add_activation_summary(relu3)
        #relu_dropout3= tf.nn.dropout(relu3, keep_prob=keep_prob)

        W4 = utils.weight_variable([3,3,Filters2,Filters2],name="W4")
        b4 = utils.bias_variable([Filters2],name="b4")
        conv4 = utils.conv2d_basic(relu3, W4, b4)

        #resnet addition
        W_Add2 = utils.weight_variable([1,1,Filters1,Filters2],name="W_Add2")
        b_Add2 = utils.bias_variable([Filters2],name="b_Add2")
        conv_Add2=utils.conv2d_strided(Addition1, W_Add2, b_Add2)
        bn_Add2=utils.batch_norm(conv_Add2, n_out=Filters2,phase_train=tf.cast(True, tf.bool),scope="bn_Add2")
        Addition2=bn_Add2+conv4

        to_decoder.append(Addition2)
#####################################Encoder Block 3 #####################################

        bn4=utils.batch_norm(Addition2, n_out=Filters2,phase_train=tf.cast(True, tf.bool),scope="bn4")
        relu4 = tf.nn.relu(bn4, name="relu4")
        #relu_dropout4= tf.nn.dropout(relu4, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu4)

        W5 = utils.weight_variable([3,3,Filters2,Filters3],name="W5")
        b5 = utils.bias_variable([Filters3],name="b5")
        conv5 = utils.conv2d_strided(relu4, W5, b5)

        bn5=utils.batch_norm(conv5, n_out=Filters3,phase_train=tf.cast(True, tf.bool),scope="bn5")
        relu5 = tf.nn.relu(bn5, name="relu5")
        if FLAGS.debug:
            utils.add_activation_summary(relu5)
        #relu_dropout5= tf.nn.dropout(relu5, keep_prob=keep_prob)

        W6 = utils.weight_variable([3,3,Filters3,Filters3],name="W6")
        b6 = utils.bias_variable([Filters3],name="b6")
        conv6 = utils.conv2d_basic(relu5, W6, b6)

        #resnet addition
        W_Add3 = utils.weight_variable([1,1,Filters2,Filters3],name="W_Add3")
        b_Add3 = utils.bias_variable([Filters3],name="b_Add3")
        conv_Add3=utils.conv2d_strided(Addition2, W_Add3, b_Add3)
        bn_Add3=utils.batch_norm(conv_Add3, n_out=Filters3,phase_train=tf.cast(True, tf.bool),scope="bn_Add3")
        Addition3=bn_Add3+conv6

        to_decoder.append(Addition3)

###################################### Bridge #####################################

        bn6=utils.batch_norm(Addition3, n_out=Filters3,phase_train=tf.cast(True, tf.bool),scope="bn6")
        relu6 = tf.nn.relu(bn6, name="relu6")
        #relu_dropout6= tf.nn.dropout(relu6, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu6)

        W7 = utils.weight_variable([3,3,Filters3,Filters4],name="W7")
        b7 = utils.bias_variable([Filters4],name="b7")
        conv7 = utils.conv2d_strided(relu6, W7, b7)

        bn7=utils.batch_norm(conv7, n_out=Filters4,phase_train=tf.cast(True, tf.bool),scope="bn7")
        relu7 = tf.nn.relu(bn7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        #relu_dropout7= tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([3,3,Filters4,Filters4],name="W8")
        b8 = utils.bias_variable([Filters4],name="b8")
        conv8 = utils.conv2d_basic(relu7, W8, b8)

        #resnet addition
        W_Add4 = utils.weight_variable([1,1,Filters3,Filters4],name="W_Add4")
        b_Add4 = utils.bias_variable([Filters4],name="b_Add4")
        conv_Add4=utils.conv2d_strided(Addition3, W_Add4, b_Add4)
        bn_Add4=utils.batch_norm(conv_Add4, n_out=Filters4,phase_train=tf.cast(True, tf.bool),scope="bn_Add4")
        Addition4=bn_Add4+conv8

#####################################Decoder Block 1 #####################################
        concat1=utils.upconv_concat(Addition4,Addition3,Filters4,'1')

        bn8=utils.batch_norm(concat1, n_out=Filters3+Filters4,phase_train=tf.cast(True, tf.bool),scope="bn8")
        relu8 = tf.nn.relu(bn8, name="relu8")
        #relu_dropout8= tf.nn.dropout(relu8, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu8)

        W9 = utils.weight_variable([3,3,Filters3+Filters4,Filters3],name="W9")
        b9 = utils.bias_variable([Filters3],name="b9")
        conv9 = utils.conv2d_basic(relu8, W9, b9)

        bn9=utils.batch_norm(conv9, n_out=Filters3,phase_train=tf.cast(True, tf.bool),scope="bn9")
        relu9 = tf.nn.relu(bn9, name="relu9")
        if FLAGS.debug:
            utils.add_activation_summary(relu9)
        #relu_dropout9= tf.nn.dropout(relu9, keep_prob=keep_prob)

        W10 = utils.weight_variable([3,3,Filters3,Filters3],name="W10")
        b10 = utils.bias_variable([Filters3],name="b10")
        conv10 = utils.conv2d_basic(relu9, W10, b10)

        #resnet addition
        W_Add5 = utils.weight_variable([1,1,Filters3+Filters4,Filters3],name="W_Add5")
        b_Add5 = utils.bias_variable([Filters3],name="b_Add5")
        conv_Add5=utils.conv2d_basic(concat1, W_Add5, b_Add5)
        bn_Add5=utils.batch_norm(conv_Add5, n_out=Filters3,phase_train=tf.cast(True, tf.bool),scope="bn_Add5")
        Addition5=bn_Add5+conv10

#####################################Decoder Block 2 #####################################
        concat2=utils.upconv_concat(Addition5,Addition2,Filters3,'2')

        bn10=utils.batch_norm(concat2, n_out=Filters2+Filters3,phase_train=tf.cast(True, tf.bool),scope="bn10")
        relu10 = tf.nn.relu(bn10, name="relu10")
        #relu_dropout10= tf.nn.dropout(relu10, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu10)

        W11 = utils.weight_variable([3,3,Filters2+Filters3,Filters2],name="W11")
        b11 = utils.bias_variable([Filters2],name="b11")
        conv11 = utils.conv2d_basic(relu10, W11, b11)

        bn11=utils.batch_norm(conv11, n_out=Filters2,phase_train=tf.cast(True, tf.bool),scope="bn11")
        relu11 = tf.nn.relu(bn11, name="relu11")
        if FLAGS.debug:
            utils.add_activation_summary(relu11)
        #relu_dropout11= tf.nn.dropout(relu11, keep_prob=keep_prob)

        W12 = utils.weight_variable([3,3,Filters2,Filters2],name="W12")
        b12 = utils.bias_variable([Filters2],name="b12")
        conv12 = utils.conv2d_basic(relu11, W12, b12)

        #resnet addition
        W_Add6 = utils.weight_variable([1,1,Filters2+Filters3,Filters2],name="W_Add6")
        b_Add6 = utils.bias_variable([Filters2],name="b_Add6")
        conv_Add6=utils.conv2d_basic(concat2, W_Add6, b_Add6)
        bn_Add6=utils.batch_norm(conv_Add6, n_out=Filters2,phase_train=tf.cast(True, tf.bool),scope="bn_Add6")
        Addition6=bn_Add6+conv12
#####################################Decoder Block 3 #####################################
        concat3=utils.upconv_concat(Addition6,Addition1,Filters2,'3')

        bn12=utils.batch_norm(concat3, n_out=Filters1+Filters2,phase_train=tf.cast(True, tf.bool),scope="bn12")
        relu12 = tf.nn.relu(bn12, name="relu12")
        #relu_dropout12= tf.nn.dropout(relu12, keep_prob=keep_prob)
        if FLAGS.debug:
            utils.add_activation_summary(relu12)

        W13 = utils.weight_variable([3,3,Filters1+Filters2,Filters1],name="W13")
        b13 = utils.bias_variable([Filters1],name="b13")
        conv13 = utils.conv2d_basic(relu12, W13, b13)

        bn13=utils.batch_norm(conv13, n_out=Filters1,phase_train=tf.cast(True, tf.bool),scope="bn13")
        relu13 = tf.nn.relu(bn13, name="relu13")
        if FLAGS.debug:
            utils.add_activation_summary(relu13)
        #relu_dropout13= tf.nn.dropout(relu13, keep_prob=keep_prob)

        W14 = utils.weight_variable([3,3,Filters1,Filters1],name="W14")
        b14 = utils.bias_variable([Filters1],name="b14")
        conv14 = utils.conv2d_basic(relu13, W14, b14)

        #resnet addition
        W_Add7 = utils.weight_variable([1,1,Filters1+Filters2,Filters1],name="W_Add7")
        b_Add7 = utils.bias_variable([Filters1],name="b_Add7")
        conv_Add7=utils.conv2d_basic(concat3, W_Add7, b_Add7)
        bn_Add7=utils.batch_norm(conv_Add7, n_out=Filters1,phase_train=tf.cast(True, tf.bool),scope="bn_Add7")
        Addition7=bn_Add7+conv14

#####################################output #####################################

        W15 = utils.weight_variable([1,1,Filters1,3],name="W15")
        b15 = utils.bias_variable([3],name="b15")
        conv15 = utils.conv2d_basic(Addition7, W15, b15)

        logits=conv15

        #final=tf.layers.conv2d(Addition7,1, (1, 1),name='final',activation=tf.nn.sigmoid,padding='same')

        #final = tf.reshape(final,[FLAGS.IMAGE_SIZE/size,FLAGS.IMAGE_SIZE/size,1])
        #sigmoid=tf.nn.sigmoid(conv15,name='sigmoid')
        #annotation_pred = tf.argmax(logits, dimension=3, name="prediction")
        annotation_pred=tf.nn.softmax(logits,dim=-1)
        annotation_pred = tf.argmax(annotation_pred, dimension=-1, name="prediction")
        #tf.expand_dims(annotation_pred, dim=3)

    return  tf.expand_dims(annotation_pred, dim=3),logits
