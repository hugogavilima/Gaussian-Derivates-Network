import json
from load_data import *
from model import *
from plotting import *
import tensorflow as tf 
import os
from numpy import sqrt as sqrt
import h5py

#Importamos los ground truth del conjunto de entrenamiento y test. 
json_train = open('JSON FILES\DTS_SG_part_A.json')
json_test = open('JSON FILES\DTS_SG_part_A_TEST.json')
paths_train = json.load(json_train)
paths_test = json.load(json_test)

#Cargamos la data 
train_GT = mLoad_GT(paths_train, n=1)
train_img = mLoad_Img(paths_train, n=1)

test_GT = mLoad_GT(paths_test, n=1)
test_img = mLoad_Img(paths_test, n=1)


#Definimos los sigmas
input_sigma = []
for i in range(12):
    input_sigma.append(0.1615*(sqrt(2))**(i))
    
#Definimos el tamaño de nuestra imagen.   
input_shape = train_img[0,:,:,:].shape

#Creamos el modelo
model = Betsy(input_shape= input_shape, input_sigmas= input_sigma, input_kernel_size=(6, 6))


#model.build_graph(input_shape).summary()
#Compliamos el modelo
model.compile(loss = GAME_loss,
              optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5), 
              metrics = [sMAE, RMSE])

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 1

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  verbose=1,
#                                                  save_freq = 20)

history = model.fit(train_img, 
                    train_GT, 
                    batch_size = batch_size, 
                    epochs = 1, 
                    validation_data=(test_img, test_GT))

print('Entrenamiento Terminado! \n')
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#model.load_weights(latest)



# for i in range(len(model.layers)):
#     try:
#         model.layers[i].deploy()
#         print('Gaussian Layer: ', i, model.layers[i].deployed, '\n')
#     except:
#         print(i, 'is no a Gaussian Layer')
        

# print('Calculando Prediccion... \n')
# predict = model.predict(test_img, batch_size=batch_size)
# print('Done!')


# print('Creando los ploting... \n')
# count_estimate(test_img, test_GT, predict, model)
# print('Done!')
    
    
    
    