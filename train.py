import json
from load_data import *
from model import *
from plotting import *
import tensorflow as tf 
import os
from numpy import sqrt as sqrt

#Importamos los ground truth del conjunto de entrenamiento y test. 
json_train = open('JSON FILES\DTS_SG_part_A.json')
json_test = open('JSON FILES\DTS_SG_part_A_TEST.json')
paths_train = json.load(json_train)
paths_test = json.load(json_test)

#Cargamos la data 
train_GT = mLoad_GT(paths_train, n=10)
train_img = mLoad_Img(paths_train, n=10)

test_GT = mLoad_GT(paths_test, n=5)
test_img = mLoad_Img(paths_test, n=5)


#Definimos los sigmas
input_sigma = []
for i in range(5):
    input_sigma.append(0.8*(sqrt(2))**(i))
    
#Definimos el tamaño de nuestra imagen.   
input_shape = train_img[0,:,:,:].shape

#Creamos el modelo
model = Betsy(input_shape= input_shape, input_sigmas= input_sigma, input_kernel_size=(9, 9))


#model.build_graph(input_shape).summary()
#Compliamos el modelo
model.compile(loss = GAME_loss,
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), 
              metrics = [sMAE(), RMSE()])

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 10

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_img, 
          train_GT, 
          batch_size = batch_size, 
          epochs = 5, 
          validation_data=(test_img, test_GT))

print('Entrenamiento Terminado! \n')
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#model.load_weights(latest)


model.layers[0].deploy()
model.layers[1].deploy()
model.layers[2].deploy()
model.layers[3].deploy()
model.layers[4].deploy()

dpy = {'G1 dpy': model.layers[0].deployed,
       'G2 dpy': model.layers[1].deployed,
       'G3 dpy': model.layers[2].deployed,
       'G4 dpy': model.layers[3].deployed,
       'G5 dpy': model.layers[4].deployed}
print(dpy)

print('Calculando Prediccion... \n')
predict = model(test_img)
print('Done!')

print('Creando los ploting... \n')
count_estimate(test_img, test_GT, predict, model)
print('Done!')
    
    
    
    