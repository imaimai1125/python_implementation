# LOAD MNIST datasets
import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# import tensorflow
import tensorflow as tf
sess = tf.InteractiveSession()
