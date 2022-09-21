import tensorflow as tf
from FTGDConvLayer import *

class Betsy(tf.keras.Model):

  def __init__(self, input_shape):
    super().__init__()
    self.gaussian1 = FTGDConvLayer(filters=16, 
                                   kernel_size = (7,7), 
                                   num_basis= 4, 
                                   order=2, 
                                   separated = True, 
                                   name = 'Gaussian1')
    self.gaussian2 = FTGDConvLayer(filters=32, 
                                   kernel_size = (7,7), 
                                   num_basis= 8, 
                                   order=2,  
                                   name = 'Gaussian2')
    self.output_layer = tf.keras.layers.Conv2D(1,1, 
                                               activation='relu',
                                               input_shape = (input_shape[0], input_shape[1], 64))

  def call(self, input):
    x = self.gaussian1(input)
    x = tf.keras.layers.Activation('relu')(x)
    x = self.gaussian2(x)
    x = tf.keras.layers.Activation('relu')(x)
    return self.output_layer(x)
  
  def build_graph(self, input_shape):
    y = tf.keras.layers.Input(shape = input_shape)
    return tf.keras.Model(inputs=[y], 
                          outputs=self.call(y))
    
    
