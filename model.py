import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from FTGDConvLayer import *

class Betsy(tf.keras.Model):

  def __init__(self, input_shape, input_sigmas, input_kernel_size):
    super().__init__()
    
    ###################################################################
    # GAUSSIAN LAYERS
    ###################################################################
    self.gaussian1 = FTGDConvLayer(filters=12, 
                                   kernel_size = input_kernel_size,   
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[0],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian1')

    self.gaussian2 = FTGDConvLayer(filters=14, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[1],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian2')
    
    self.gaussian3 = FTGDConvLayer(filters=16, 
                                   kernel_size = input_kernel_size,  
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[2],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian3')
    
    self.gaussian4 = FTGDConvLayer(filters=20, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[3],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian4')
    
    self.gaussian5 = FTGDConvLayer(filters=64, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[4],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian5')

    self.gaussian6 = FTGDConvLayer(filters=32, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[5],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian6')
    
    self.gaussian7 = FTGDConvLayer(filters=64, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[6],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian7')
    
    self.gaussian8 = FTGDConvLayer(filters=32, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[7],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian8')
        
    self.gaussian9 = FTGDConvLayer(filters=20, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[8],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian9')
    
    self.gaussian10 = FTGDConvLayer(filters=16, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[9],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian10')
    
    self.gaussian11 = FTGDConvLayer(filters=8, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[10],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian11')
    
    self.gaussian12 = FTGDConvLayer(filters=1, 
                                   kernel_size = input_kernel_size, 
                                   num_basis= 1, 
                                   order=2, 
                                   separated = False,
                                   trainability=[False, False, False],
                                   sigma_init= input_sigmas[11],
                                   random_init=False, 
                                   use_bias=False,
                                   name = 'Gaussian12')
    
    ###################################################################
    # NORMALIZATION LAYERS
    ###################################################################
    
    self.BN_1 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_1')
    self.BN_2 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_2')
    self.BN_3 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_3')
    self.BN_4 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_4')
    self.BN_5 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_5')
    self.BN_6 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_6')
    self.BN_7 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_7')
    self.BN_8 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_8')
    self.BN_9 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_9')
    self.BN_10 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_10')
    self.BN_11 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_11')
    self.BN_12 = tf.keras.layers.BatchNormalization(axis=-1, name = 'BN_12')
    
    ###################################################################
    # OUTPUT LAYERS
    ###################################################################
    self.output_layer = tf.keras.layers.Conv2D(1,1, 
                                               activation='relu',
                                               input_shape = (input_shape[0], input_shape[1], 32))
    
    #
       
  def call(self, input):
    x = self.gaussian1(input)
    #x = self.BN_1(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian2(x)
    #x = self.BN_2(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian3(x) 
    #x = self.BN_3(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian4(x)
    #x = self.BN_4(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian5(x) 
    #x = self.BN_5(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian6(x) 
    #x = self.BN_6(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian7(x) 
    #x = self.BN_7(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian8(x) 
    #x = self.BN_8(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian9(x) 
    #x = self.BN_9(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian10(x)
    #x = self.BN_10(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian11(x) 
    #x = self.BN_11(x)
    #x = tf.keras.activations.relu(x)
    x = self.gaussian12(x)
    #x = self.BN_12(x)
    #x = tf.keras.activations.relu(x)
    return x
   

  def build_graph(self, input_shape):
    y = tf.keras.layers.Input(shape = input_shape)
    return tf.keras.Model(inputs=[y], 
                          outputs=self.call(y))

"""
##########################################################################################
METRICAS:
    En este apartado, definimos las metricas usadas durante el entrenamiento
########################################################################################## 
"""

def sMAE(y_true, y_pred):
  res2 = tf.constant(0, dtype=np.float32)
    
  for i in range(len(y_pred)):
    bb = tf.math.abs(tf.math.reduce_sum(y_true[i]) - tf.math.reduce_sum(y_pred[i]))
    res2 = tf.math.add(res2, bb) 
      
  values = tf.math.divide(res2, tf.cast(len(y_pred), tf.float32))
    
  return values
 
 
def RMSE(y_true, y_pred):
  res2 = tf.constant(0, dtype=np.float32)
    
  for i in range(len(y_pred)):
    bb = tf.math.abs(tf.math.reduce_sum(y_true[i]) - tf.math.reduce_sum(y_pred[i]))
    res2 = tf.math.add(res2, tf.math.square(bb)) 
      
  values = tf.math.divide(res2, tf.cast(len(y_pred), tf.float32))
  values = tf.math.sqrt(values)
    
  return values 
  
  
"""
##########################################################################################
FUNCION DE PERDIDA:
    En este apartado, definimos lafuncion de perdida, como sus funciones auxiliares. 
########################################################################################## 
"""

def adjust_dim(array):
    if array.shape[0]%2 != 0:
        array = tf.pad(array, tf.constant([[0, 1], [0, 0], [0, 0]]), "CONSTANT")
    if array.shape[1]%2 != 0:
        array = tf.pad(array, tf.constant([[0, 0], [0, 1], [0, 0]]), "CONSTANT")
    return array



def four_split_tf(mtf):
    shape = mtf.shape
    x, y = shape[0], shape[1]
    
    x2, y2 = x//2, y//2
    
    #mtf4 = tf.constant(0, shape = [4, x2, y2, 1], dtype= tf.float32)
    sumi = tf.concat([tf.expand_dims(mtf[0:x2, 0:y2, :], axis=0),
                      tf.expand_dims(mtf[0:x2, y2:, :], axis=0),
                      tf.expand_dims(mtf[x2:, 0:y2, :], axis=0),
                      tf.expand_dims(mtf[x2:, y2:, :], axis=0)], 0)
    #mtf4 = mtf4 + sumi
    
    return sumi


def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = tf.math.abs(-tf.math.reduce_sum(density) + tf.math.reduce_sum(gt))
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
    res2 = res2 + (GAME_recursive(preds[i], gts[i], 0, 5))
  return tf.math.divide(res2, tf.cast(len(gts), tf.float32))





