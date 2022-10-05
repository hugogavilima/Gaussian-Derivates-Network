import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from FTGDConvLayer import *

class Betsy(tf.keras.Model):

  def __init__(self, input_shape, input_sigmas, input_kernel_size):
    super().__init__()
    self.gaussian1 = FTGDConvLayer(filters=12, 
                                   kernel_size = input_kernel_size,   
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, True, True],
                                   sigma_init= input_sigmas[0],
                                   random_init=False, 
                                   use_bias=True,
                                   name = 'Gaussian1')

    self.gaussian2 = FTGDConvLayer(filters=14, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, True, True],
                                   sigma_init= input_sigmas[1],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian2')
    
    self.gaussian3 = FTGDConvLayer(filters=16, 
                                   kernel_size = input_kernel_size,  
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, True, True],
                                   sigma_init= input_sigmas[2],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian3')
    
    self.gaussian4 = FTGDConvLayer(filters=20, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, True, True],
                                   sigma_init= input_sigmas[3],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian4')
    
    self.gaussian5 = FTGDConvLayer(filters=64, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 4, 
                                   order=2, 
                                   separated = True,
                                   trainability=[False, True, True],
                                   sigma_init= input_sigmas[4],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian5')
    
    self.output_layer = tf.keras.layers.Conv2D(1,1, 
                                               activation='relu',
                                               input_shape = (input_shape[0], input_shape[1], 64))
       
    self.sMAE = sMAE()
    self.RMSE = RMSE()


  def call(self, input):
    x = self.gaussian1(input)
    x = self.gaussian2(x)
    x = self.gaussian3(x)
    x = self.gaussian4(x)
    x = self.gaussian5(x) 
    return self.output_layer(x)
   

  def get_loss(self, train_image, test_GT):
    train_pred_GT = self.call(train_image)
    return GAME_loss(train_pred_GT, test_GT)
  
  def get_sMAE(self, train_image, test_GT):
    
    train_pred_GT = self.call(train_image)
    self.sMAE.reset_state()
    self.sMAE.update_state(test_GT, train_pred_GT)
    
    return self.sMAE.result()

  def build_graph(self, input_shape):
    y = tf.keras.layers.Input(shape = input_shape)
    return tf.keras.Model(inputs=[y], 
                          outputs=self.call(y))
  
class sMAE(tf.keras.metrics.Metric):

  def __init__(self, name= 'sMAE', **kwargs):
    super(sMAE, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    
    res2 = tf.constant(0, dtype=np.float32)
    for i in range(len(y_pred)):
      res2 = res2 + (GAME_recursive(y_pred[i], y_true[i], 0, 0))
      
    values = tf.math.divide(res2, tf.cast(len(y_pred), tf.float32))
    values = tf.cast(values, self.dtype)
    
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, values.shape)
      #values = tf.multiply(values, sample_weight)
      
    self.true_positives.assign_add(values)

  def result(self):
    return self.true_positives

class RMSE(tf.keras.metrics.Metric):

  def __init__(self, name= 'RMSE', **kwargs):
    super(RMSE, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    res2 = tf.constant(0, dtype=np.float32)
    for i in range(len(y_pred)):
      bb = GAME_recursive(y_pred[i], y_true[i], 0, 0)
      res2 = res2 + tf.math.square(bb)
      
    values = tf.math.divide(res2, tf.cast(len(y_pred), tf.float32))
    values = tf.math.sqrt(values)
    values = tf.cast(values, self.dtype)
    
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, values.shape)
      #values = tf.multiply(values, sample_weight)
      
    self.true_positives.assign_add(values)

  def result(self):
    return self.true_positives



 
def adjust_dim(array):
    if array.shape[0]%2 != 0:
        array = tf.pad(array, tf.constant([[0, 1], [0, 0], [0, 0]]), "CONSTANT")
    if array.shape[1]%2 != 0:
        array = tf.pad(array, tf.constant([[0, 0], [0, 1], [0, 0]]), "CONSTANT")
    return array


def four_split_tf(mtf):
    shape = mtf.shape
    x, y = shape[0], shape[1]
    
    x2, y2 = x//2, y//2, 
    
    mtf4 = tf.constant(0, shape = [4, x2, y2, 1], dtype= tf.float32)
    sumi = tf.concat([tf.expand_dims(mtf[0:x2, 0:y2, :], axis=0),
                      tf.expand_dims(mtf[0:x2, y2:, :], axis=0),
                      tf.expand_dims(mtf[x2:, 0:y2, :], axis=0),
                      tf.expand_dims(mtf[x2:, y2:, :], axis=0)], 0)
    mtf4 = mtf4 + sumi
    
    return mtf4


def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = tf.math.abs(tf.math.reduce_sum(density) - tf.math.reduce_sum(gt))
        return game
    
    else:
        density = adjust_dim(density)
        gt = adjust_dim(gt)
        
        density_slice = four_split_tf(density)
        gt_slice = four_split_tf(gt)
        
               
        currentLevel = currentLevel + 1 
        res = tf.constant(0, dtype=np.float32)
        for a in range(4):
            res = res + (GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
        
        return res

def GAME_loss(preds, gts):
  res2 = tf.constant(0, dtype=np.float32)
  for i in range(len(gts)):
    res2 = res2 + (GAME_recursive(preds[i], gts[i], 0, 1))
  return tf.math.divide(res2, tf.cast(len(gts), tf.float32))





