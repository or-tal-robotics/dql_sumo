#!/usr/bin/env python

import tensorflow as tf


class ImageTransformer():
    def __init__(self, image_size):
        with tf.variable_scope("image_transformer", reuse=tf.AUTO_REUSE):
            self.input_state = tf.placeholder(shape = [480,640,3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.resize_images(self.output, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
            
    def transform(self, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, feed_dict = {self.input_state: state})