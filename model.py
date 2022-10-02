import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
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
    self.train_op = tf.keras.optimizers.Adam(learning_rate = 0.01)
    
    self.sMAE = sMAE()
    self.RMSE = RMSE()


  def call(self, input):
    x = self.gaussian1(input)
    x = tf.keras.layers.Activation('relu')(x)
    x = self.gaussian2(x)
    x = tf.keras.layers.Activation('relu')(x)
    return self.output_layer(x)

  def get_loss(self, train_image, test_GT):
    train_pred_GT = self.call(train_image)
    return GAME_loss(train_pred_GT, test_GT)
  
  def get_sMAE(self, train_image, test_GT):
    
    train_pred_GT = self.call(train_image)
    self.sMAE.reset_state()
    self.sMAE.update_state(test_GT, train_pred_GT)
    
    return self.sMAE.result()

  def get_grad(self, train_image, test_GT):
    with tf.GradientTape() as tape:
        tape.watch(self.gaussian1.variables)
        tape.watch(self.gaussian2.variables)
        tape.watch(self.output_layer.variables)
        L = self.get_loss(train_image, test_GT)
        g = tape.gradient(L, self.gaussian1.variables + self.gaussian2.variables)
    return g 
  
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
    res2 = res2 + (GAME_recursive(preds[i], gts[i], 0, 0))
  return tf.math.divide(res2, tf.cast(len(gts), tf.float32))



#Corriegele, esta funcion debe tambien tomar el path de la imagen, y con esto hacerle el grafico. 
#Esta función ebe ser más propia para identificar el path de la imagen. 
def plotting(i, test_img):
    font = {'color':  'black','weight': 'normal','size': 16}

    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(test_img[i,:,:,0], cmap = 'gray')
    ax1.axis('off')
    ax1.set_title('Imagen Real', fontdict=font)
    ax1.text(0.1, -0.1, 'right bottom',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax1.transAxes, )

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('GT', fontdict=font)
    ax2.imshow(test_GT[i,:,:,0], interpolation='gaussian')
    ax2.axis('off')

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Densidad Estimada', fontdict=font)
    ax3.imshow(tt[i,:,:,0], interpolation='gaussian')
    ax3.axis('off')


    fig.savefig('plots/test_' + str(i) + '.png')

  



