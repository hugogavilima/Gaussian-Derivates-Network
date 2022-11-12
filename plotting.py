import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import pandas as pd
import os 
from model import *
from progressbar import progressbar 


def count_estimate(test_img, test_gt, predict, name, type):
    
    root = r'C:\Users\Usuario\Documents\results'
    path = os.path.join(root, name)
    loss = tf.keras.losses.MeanAbsoluteError()
    
    try:
        os.mkdir(path)
    except:
        None
    
    N = len(test_gt)
    resume = pd.DataFrame({'GT':[], 'Pred':[], 'Loss':[], 'MAE': []})
    progress = progressbar.ProgressBar()
    for i in progress(range(N)):
        IMG = test_img[i,:,:,0]
        GT = test_gt[i,:,:,0]
        PRED = predict[i,:,:,0]
        
        
        est_count = tf.math.reduce_sum(PRED)
        GT_count = tf.math.reduce_sum(GT)
        est_loss = loss(GT, PRED).numpy()
        est_MAE =  tf.math.abs(tf.math.reduce_sum(GT) - tf.math.reduce_sum(PRED))
        
        resume.loc[i, 'GT'] = np.float32(GT_count)
        resume.loc[i, 'Pred'] = np.float32(est_count)
        resume.loc[i, 'Loss'] = np.float32(est_loss)
        resume.loc[i, 'MAE'] = np.float32(est_MAE)       
        
        
        plotting_testing(IMG, GT, PRED, GT_count, est_count, est_loss, est_MAE, i, path, type) 
    
    resume.to_excel(os.path.join(path, name + '.xlsx'))  
                   
      
                
def plotting_testing(test_img, test_GT, predict, GT_count, est_count, est_loss, est_MAE, i, path, type):
    font = {'color':  'black','weight': 'normal','size': 16}

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(test_img, cmap = 'gray')
    ax1.axis('off')
    ax1.set_title('Imagen Real', fontdict=font)
    ax1.text(0.5, -0.1, 
            'Count: %.2f;       Estimate Count: %.2f;    Loss: %.2f;     MAE: %.2f' % (np.float32(GT_count), np.float32(est_count), np.float32(est_loss), np.float32(est_MAE)),
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax1.transAxes, )

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('GT', fontdict=font)
    ax2.imshow(test_GT, interpolation='gaussian')
    ax2.axis('off')

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Densidad Estimada', fontdict=font)
    ax3.imshow(predict, interpolation='gaussian')
    ax3.axis('off')
    
    fig_path = os.path.join(path, type)
    try:
        os.mkdir(fig_path)
    except:
        None
    fig.savefig(os.path.join(fig_path, str(i) + '.png'));
    plt.close(fig)



def deploy_layers(model):
  layers = model.layers
  for ly in layers:
    try:
        ly.deploy()
        print(ly, ' deploy: ', ly.deployed)
    except:
        print(ly, 'is no a Gaussian Layer')
        

def train_layers(model):
  layers = model.layers
  for ly in layers:
    try:
        ly.to_train()
        print(ly, ' deploy: ', ly.deployed)
    except:
        print(ly, 'is no a Gaussian Layer')
        
