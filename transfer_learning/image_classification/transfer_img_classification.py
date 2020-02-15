import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from random import shuffle
import time
import cv2

'''
Transfer learning for image classification
Transfer from a dog recognition model to a cat recognition model
The scripts are implemented with Python 3.7 and the model is implemented with Tensorflow 1.14.0
Image data are from ImageNet dataset
'''

path = "./images/ILSVRC/Data/CLS-LOC/train/"  # Path of ImageNet dataset

tf.debugging.set_log_device_placement(True)  # Find out which devices your operations and tensors are assigned to

def read_data(path, train_c1_list, train_c1_interval, train_c0_list, train_c0_interval, test_c1_list, test_c1_interval, test_c0_list, test_c0_interval, img_height, img_width):

  def read_features_and_labels(data_class_list, label, interval):
      X_data=[]
      y_data = []

      for data_class in data_class_list:
          data_fnames = os.listdir(path + data_class)
          filter=lambda a, b: data_fnames[a: b]
          data_img_path = [os.path.join(path + data_class, fname) for fname in filter(interval[0], interval[1])]
          for img_path in data_img_path:
              img = cv2.resize(mpimg.imread(img_path),
                               (img_height, img_width))  # Read an image and crop its size to img_height*img_width
              X_data.append(img / 255)  # Normalized each pixel value to between 0 and 1
              y_data.append(label)
      return X_data, y_data

  X1_train, y1_train = read_features_and_labels(train_c1_list, 1, train_c1_interval)
  X0_train, y0_train = read_features_and_labels(train_c0_list, 0, train_c0_interval)
  X1_train.extend(X0_train)  # Add class 0 data into a class 1 data list
  X_train = X1_train
  y1_train.extend(y0_train)
  y_train = y1_train

  X1_test, y1_test = read_features_and_labels(test_c1_list, 1, test_c1_interval)
  X0_test, y0_test = read_features_and_labels(test_c0_list, 0, test_c0_interval)
  X1_test.extend(X0_test)  # Add class 0 data into a class 1 data list
  X_test = X1_test
  y1_test.extend(y0_test)
  y_test = y1_test

  return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def build_graph(img_height=32, img_width=32, dict_config={}):
  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    with tf.device('/device:GPU:0'):
        # Build a 2d-CNN model
        n_channels = 3  # RGB images
        n_classes = 2  # Two classes: 1 for dogs(or cats), and 0 for others

        X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, n_channels], name='X')
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        is_train = tf.placeholder_with_default(False, (), name='is_train')

        fil_height = 5
        fil_width = 5
        stddev = 0.1
        batch_stride = 1
        vertical_stride = 2
        horizontal_stride = 2
        channel_stride = 1
        padding = "SAME"
        dropout_rate = 0.2
        seed = 3

        n_fil1_outputs = 64
        filters1 = tf.Variable(tf.random.truncated_normal(shape=(fil_height, fil_width, n_channels, n_fil1_outputs), stddev=stddev, seed=seed), name="filters1")

        conv1 = tf.nn.conv2d(X, filter=filters1, strides=[batch_stride, vertical_stride, horizontal_stride, channel_stride], padding=padding)
        #conv1 = tf.layers.batch_normalization(conv1, training=is_train, name="bn_conv1")
        conv1 = tf.nn.selu(conv1)  # SELU activation function
        conv1_dropout = tf.nn.dropout(conv1, rate=dropout_rate)  # Use rate instead of keep_prob

        n_fil2_outputs = 128
        filters2 = tf.Variable(tf.random.truncated_normal(shape=(fil_height, fil_width, n_fil1_outputs, n_fil2_outputs), stddev=stddev), name="filters2")

        conv2 = tf.nn.conv2d(conv1_dropout, filter=filters2, strides=[batch_stride, vertical_stride, horizontal_stride, channel_stride], padding=padding)
        conv2 = tf.nn.selu(conv2)
        conv2_dropout = tf.nn.dropout(conv2, rate=dropout_rate)  # Use rate instead of keep_prob

        n_fil3_outputs = 128
        filters3 = tf.Variable(tf.random.truncated_normal(shape=(fil_height, fil_width, n_fil1_outputs, n_fil3_outputs), stddev=stddev), name="filters3")

        conv3 = tf.nn.conv2d(conv2_dropout, filter=filters3, strides=[batch_stride, vertical_stride, horizontal_stride, channel_stride], padding=padding)
        #conv3 = tf.layers.batch_normalization(conv3, training=is_train, name="bn_conv3")
        conv3 = tf.nn.selu(conv3)
        conv3_dropout = tf.nn.dropout(conv3, rate=dropout_rate)  # Use rate instead of keep_prob

        temp=1  # Used for flattening
        for shape in conv3_dropout.shape[1:]:
          temp *= int(shape)

        flatten = tf.reshape(conv3_dropout, [-1, temp])

        # Fully connected layer
        reg_lambda = 0.001  # Used for regularization

        n_h1_outputs = int(flatten.shape[1]) // 2
        h1_outputs = tf.layers.dense(flatten, n_h1_outputs, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=seed), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_lambda), name="dense1")
        h1_outputs = tf.layers.batch_normalization(h1_outputs, training=is_train, name="bn_h1")
        h1_outputs = tf.nn.selu(h1_outputs)
        h1_outputs_dropout = tf.nn.dropout(h1_outputs, rate=dropout_rate)  # Use rate instead of keep_prob

        n_h2_outputs = n_h1_outputs // 2
        h2_outputs = tf.layers.dense(h1_outputs_dropout, n_h2_outputs, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=seed), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_lambda), name="dense2")
        h2_outputs = tf.layers.batch_normalization(h2_outputs, training=is_train, name="bn_h2")
        h2_outputs=tf.nn.selu(h2_outputs)
        h2_outputs_dropout = tf.nn.dropout(h2_outputs, rate=dropout_rate)  # Use rate instead of keep_prob

        n_h3_outputs = n_classes
        h3_outputs = tf.layers.dense(h2_outputs_dropout, n_h3_outputs, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=seed), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_lambda), name="dense3")
        h3_outputs = tf.layers.batch_normalization(h3_outputs, training=is_train, name="bn_h3")
        logits= tf.identity(h3_outputs, name="logits")

        learning_rate = 0.001
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")

        if dict_config and dict_config['trainable_scope']:
            # Scripts for only updating a part of weights of the model
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dict_config['trainable_scope'])
            var_list = train_vars
        else:
            var_list = None

        train_op = optimizer.minimize(loss, var_list=var_list, name="train_op")

        correct_prediction = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

  return g

# def load_trained_source_graph(load_ckpt_meta):
#   with tf.device('/device:GPU:0'):
#       # Load the trained model
#       saver = tf.train.import_meta_graph(load_ckpt_meta)
#       # Access as the default graph
#       g = tf.get_default_graph()
#       return g, saver

def train_model(X_train, y_train, X_test, y_test, img_height=32, img_width=32, source=True, dict_config={}):
  # Build a model(graph) for training
  g = build_graph(img_height, img_width, dict_config)

  # Settings before training the model
  n_epoch = 500
  batch_size = 5
  n_batch = int(np.ceil(X_train.shape[0]/batch_size))

  # Start the Tensorflow Session
  with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    #print("sess is tf.get_default_session(): ",sess is tf.get_default_session())

    now = time.strftime("%Y%m%d%H%M", time.localtime())

    # Get Tensors and some operations from the graph
    train_op = g.get_operation_by_name("train_op")
    filters1 = g.get_tensor_by_name("filters1:0")
    filters2 = g.get_tensor_by_name("filters2:0")
    filters3 = g.get_tensor_by_name("filters3:0")
    is_train = g.get_tensor_by_name("is_train:0")
    loss = g.get_tensor_by_name("loss:0")
    logits = g.get_tensor_by_name("logits:0")
    accuracy = g.get_tensor_by_name("accuracy:0")

    # https://timodenk.com/blog/tensorflow-batch-normalization/
    update_batch_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # To update parameters of batch normalization
    
    # Initialize the Variables or load a trained model
    if not dict_config:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
    else:
      sess.run(tf.global_variables_initializer())
      reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dict_config['init_scope'])  #"filters[12]|" # Reuse trained weights of filter1 and filter2
      reuse_vars_list = [var for var in reuse_vars]
      saver = tf.train.Saver(reuse_vars_list)  # saver to restore the original model
      saver.restore(sess, dict_config['ckpt'])

    indices = np.arange(X_train.shape[0])
    list_avg_loss_batch = []
    list_accuracy_train = []
    list_avg_loss_test = []
    list_accuracy_test = []
    shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    for epoch in range(n_epoch):
        print("epoch=",epoch)
        list_loss_batch=[]
        for i in range(n_batch):
          X_batch = X_train[i * batch_size:(i + 1) * batch_size]
          y_batch = y_train[i * batch_size:(i + 1) * batch_size]
          sess.run([train_op, update_batch_ops], feed_dict={"X:0": X_batch, "y:0": y_batch, is_train: True})
          loss_batch, output_update_batch_ops = sess.run([loss, update_batch_ops], feed_dict={"X:0": X_batch, "y:0": y_batch, is_train: True})
          list_loss_batch.append(loss_batch)
        print("list_loss_batch=",list_loss_batch)
        list_avg_loss_batch.append(np.mean(list_loss_batch))

        accuracy_train, output_update_batch_ops = sess.run([accuracy, update_batch_ops], feed_dict={"X:0": X_train, "y:0": y_train, is_train: True})
        print("accuracy_train=", accuracy_train)
        list_accuracy_train.append(accuracy_train)

        loss_test = sess.run(loss, feed_dict={"X:0": X_test, "y:0": y_test})
        print("loss_test=",loss_test)
        list_avg_loss_test.append(loss_test)

        accuracy_test = sess.run(accuracy, feed_dict={"X:0": X_test, "y:0": y_test})
        print("accuracy_test=", accuracy_test)
        list_accuracy_test.append(accuracy_test)

        # Save model checkpoint file for every 100 epochs
        if (epoch+1)%100 == 0:
          if source == True:
            subfolder = "source_model"
          else:
            subfolder = "target_model"
          save_path = saver.save(sess, subfolder + '/' + now + '_model_epoch' + str(epoch + 1) + '.ckpt')
          print("Saved a model!")
    
    
    # Plot loss after training
    if source == True:
       subfolder = "source_plot"
    else:
       subfolder = "target_plot"
    
    plt.plot(list_avg_loss_batch, label="loss_batch", color='blue')
    plt.plot(list_avg_loss_test, label="loss_test", color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Average loss for each epoch')
    plt.legend()
    plt.savefig(subfolder + '/' + now + '_loss_epochs.png')
    plt.clf()

    # Plot accuracy after training
    plt.plot(list_accuracy_train, label="accuracy_train", color='blue')
    plt.plot(list_accuracy_test, label="accuracy_test", color='green')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy for each epoch')
    plt.savefig(subfolder + '/' + now + '_accuracy_epochs.png')
    plt.clf()

    # Plot false recognized images
    logits_result = sess.run(logits, feed_dict={"X:0": X_test})
    #print("logits_result=", logits_result)
    list_false_images = []
    list_false_indices = []
    list_n_false = [0] * 2  # list_false[0]: number of false positives; list_false[1]: number of false negatives
    for i in range(X_test.shape[0]):
      pred_class = np.where(logits_result[i] == max(logits_result[i]))
      if pred_class[0][0] != int(y_test[i]):
        list_false_images.append(X_test[i])
        list_false_indices.append(pred_class[0][0])
        list_n_false[int(y_test[i])] += 1
    print("Number of false recognition on test dataset:", len(list_false_images))
    print("list_n_false=",list_n_false)

    # Plot false recognized images in a window
    img_i = 0
    frame_height = 10
    frame_width = 10
    n_plot_windows = int(np.ceil(len(list_false_images)/(frame_height*frame_width)))
    img_h = 16
    img_w = 16
    for j in range(n_plot_windows):
      fig = plt.figure(figsize=(img_h, img_w))
      if j < n_plot_windows-1:
        for i in range(1, frame_height*frame_width+1):
          a = fig.add_subplot(frame_height, frame_width, i)
          a.set_title(list_false_indices[img_i])
          plt.imshow(list_false_images[img_i])
          img_i += 1
        plt.show()
      else:
        n_rest_img = len(list_false_images)-j*(frame_height*frame_width)
        for i in range(1, n_rest_img+1):
          a = fig.add_subplot(np.ceil(n_rest_img/frame_height), frame_width, i)
          a.set_title(list_false_indices[img_i])
          plt.imshow(list_false_images[img_i])
          img_i += 1
        plt.show()
