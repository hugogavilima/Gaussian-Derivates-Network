import json
import h5py
from load_data import *
from model import *
from scipy import io 
import tensorflow as tf 
import os 
from scipy import io 

#Importamos los ground truth del conjunto de entrenamiento y test. 
json_train = open('JSON FILES\DTS_SG_part_A.json')
json_test = open('JSON FILES\DTS_SG_part_A_TEST.json')
paths_train = json.load(json_train)
paths_test = json.load(json_test)

#Cargamos la data 
train_GT = mLoad_GT(paths_train, n=200)
train_img = mLoad_Img(paths_train, n=200)

test_GT = mLoad_GT(paths_test, n=100)
test_img = mLoad_Img(paths_test, n=100)

input_shape = train_img[0,:,:,:].shape
model = Betsy(input_shape)

model.compile(loss = tf.keras.losses.MeanSquaredError(), 
              optimizer  = tf.keras.optimizers.Adam(learning_rate = 0.01), 
              metrics=['mean_squared_error'])

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = model.fit(train_img, 
                    train_GT, 
                    batch_size = 30, 
                    epochs = 5, 
                    validation_data=(test_img, test_img), 
                    callbacks=[cp_callback])



    
    
    
    