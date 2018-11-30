import os
import cv2
import json
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys

from tensorflow.contrib import rnn as tf_rnn
from VGGModel import VGGModel

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


class DSA_RCNN:

    def __init__(self, verbose=True):
    ''' Build DSA RCNN '''
    # tf Graph input
    x = tf.placeholder("float", [None, n_frames ,n_detection, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float",[None])

    # Define weights
    weights = {
        'em_obj': tf.Variable(tf.random_normal([n_input,n_att_hidden], mean=0.0, stddev=0.01)), # 1 x 4096 x 256 x 1
        'em_img': tf.Variable(tf.random_normal([n_input,n_img_hidden], mean=0.0, stddev=0.01)),
        'att_w': tf.Variable(tf.random_normal([n_att_hidden, 1], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([n_att_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01))
    }
    biases = {
        'em_obj': tf.Variable(tf.random_normal([n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_img_hidden], mean=0.0, stddev=0.01)),
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01))
    }

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden,initializer= tf.random_normal_initializer(mean=0.0,stddev=0.01),use_peepholes = True,state_is_tuple = False)
    # using dropout in output of LSTM
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1 - keep[0])
    ## init LSTM parameters
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    # init loss 
    loss = 0.0  
    # Mask 
    zeros_object = tf.to_float(tf.not_equal(tf.reduce_sum(tf.transpose(x[:,:,1:n_detection,:],[1,2,0,3]),3),0)) # frame x n x b
    # Start creat graph
    for i in range(n_frames):
      with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        # input features (Faster-RCNN fc7)
        X = tf.transpose(x[:,i,:,:], [1, 0, 2])  # permute n_steps and batch_size (n x b x h)
        # frame embedded
        image = tf.matmul(X[0,:,:],weights['em_img']) + biases['em_img'] # 1 x b x h
        # object embedded
        n_object = tf.reshape(X[1:n_detection,:,:], [-1, n_input]) # (n_steps*batch_size, n_input)
        n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj'] # (n x b) x h
        n_object = tf.reshape(n_object,[n_detection-1,batch_size,n_att_hidden]) # n-1 x b x h
        n_object = tf.multiply(n_object,tf.expand_dims(zeros_object[i],2))

        # object attention
        brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [n_detection-1,1,1]) # n x h x 1
        image_part = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['att_ua'], 0), [n_detection-1,1,1])) + biases['att_ba'] # n x b x h
        e = tf.tanh(tf.matmul(h_prev,weights['att_wa'])+image_part) # n x b x h
        # the probability of each object
        alphas = tf.multiply(tf.nn.softmax(tf.reduce_sum(tf.matmul(e,brcst_w),2),0),zeros_object[i])
        # weighting sum
        attention_list = tf.multiply(tf.expand_dims(alphas,2),n_object)
        attention = tf.reduce_sum(attention_list,0) # b x h
        # concat frame & object
        fusion = tf.concat([image,attention],1)

        with tf.variable_scope("LSTM") as vs:
            outputs,istate = lstm_cell_dropout(fusion,istate)
            lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
        # save prev hidden state of LSTM
        h_prev = outputs
        # FC to output
        pred = tf.matmul(outputs,weights['out']) + biases['out'] # b x n_classes
        # save the predict of each time step
        if i == 0:
            soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            all_alphas = tf.expand_dims(alphas,0)
        else:
            temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            soft_pred = tf.concat([soft_pred,temp_soft_pred],1)
            temp_alphas = tf.expand_dims(alphas,0)
            all_alphas = tf.concat([all_alphas, temp_alphas],0)

        # positive example (exp_loss)
        pos_loss = -1*tf.multiply(tf.exp(-(n_frames-i-1)/20.0),-1*tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
        # negative example
        neg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = pred) # Softmax loss
        temp_pos_loss = tf.add(tf.multiply(pos_loss, y[:,3]), tf.multiply(pos_loss, y[:,2]))
        temp_neg_loss = tf.add(tf.multiply(neg_loss, y[:,1]), tf.multiply(neg_loss, y[:,0]))
        temp_loss = tf.reduce_mean(tf.add(temp_pos_loss, temp_neg_loss))
        # temp_loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss,y[:,1]),tf.multiply(neg_loss,y[:,0])))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        loss = tf.add(loss, temp_loss)u
        
    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss/n_frames) # Adam Optimizer

    ''' Initialize VGG network for feature extraction '''
    VGG_model = VGGModel(verbose=True)
    print("VGG model built...")
    self.VGG_model = VGG_model
    self.x = x
    self.y = y
    self.keep = keep
    self.optimizer = optimizer
    self.loss = loss
    self.lstm_variables = lstm_variables 
    self.soft_pred = soft_pred
    self.all_alphas = all_alphas
    print("DSA RCNN Model built...")



''' Given a frame and list of obj coordinates, crop objects in frame, then extracts object features ''' 
def crop_object_features(frame, coord_list, network=self.VGG_model):
    obj_list = []

    # Append full frame first
    frame = cv2.resize(frame, (224, 224))
    full_frame_feat = network.extract_feature(frame)
    obj_list.append(full_frame_feat)

    if (len(coord_list) <= 9):
        for i in range(0, len(coord_list)):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            obj_list.append(obj_feat)
        
        # If less than 10 objects detected
        for j in range(len(coord_list), 9):
            obj_feat = np.zeros_like(obj_list[len(coord_list)])
            obj_list.append(obj_feat)
    else:
        for i in range(0, 9):
            cropped = frame[coord_list[i]['y']:coord_list[i]['y']+coord_list[i]['height'],coord_list[i]['x']:coord_list[i]['x']+coord_list[i]['width']]
            obj_feat = network.extract_feature(cropped)
            obj_list.append(obj_feat)
            # print("Object list length: {}".format(len(obj_list)))

    return np.asarray(obj_list)


''' Output cropped images for each frame in clip given videopath, path to corresponding JSON  '''
def objects_from_clip(path, start, finish, network):
    obj_frames = []
    ctr = start
    while (ctr <= finish):
        # Read JSON file with
        json_path = path + ('{}.json'.format(ctr))
        with open(json_path) as f:
            json_clip = json.load(f)

        frame = cv2.imread(path + ('{}.jpg'.format(ctr)))
        obj_coords = json_clip["objects"]
        frame_objects = crop_object_features(frame, obj_coords, network)
        obj_frames.append(frame_objects)
        ctr +=1

    return np.asarray(obj_frames)


''' Given VGG and DSA-RNN models initialized and called, extract features on each frame of a given path, return labels'''
def predict(file_path, start, finish, model_path='./model/', network=self.VGG_model):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path + 'final_model')
    print ("model restored success")

    soft_pred = self.soft_pred
    
    # Get feature vector of shape 100 x 10 x 4096 x 1
    X_raw = objects_from_clip(file_path, start, finish, network)
    # Squeeze down to vector of shape 100 x 10 x 4096
    test_X = np.squeeze(X_raw, -1)
    y_dummies = []

    # Create dummy y vector of size 100 x 5
    for i in range(test_X.shape[0]):
        y_dummies.append(np.zeros(5))
    y_dummies = np.asarray(y_dummies)

    feed_dict = {x: test_X, y: y_dummies ,keep: [0.0]}
    classification = sess.run(soft_pred, feed_dict)

    return classification
