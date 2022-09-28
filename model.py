import tensorflow as tf
import numpy as np
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
    
  def 
    


 
 
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
    
    mtf4 = tf.constant(np.array([mtf[0:x2, 0:y2, :],
                                    mtf[0:x2, y2:, :],
                                    mtf[x2:, 0:y2, :],
                                    mtf[x2:, y2:, :]]), shape = [4, x2, y2, 1])
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
        res = tf.Variable(0, dtype=np.float32)
        for a in range(4):
            res.assign_add(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
        
        return res

def GAME_metric(preds, gts, l = 0):
	res2 = tf.Variable(0, dtype=np.float32)
	for i in range(len(gts)):
		res2.assign_add(GAME_recursive(preds[i], gts[i], 0, l))
	return tf.math.divide(res2, len(gts))


