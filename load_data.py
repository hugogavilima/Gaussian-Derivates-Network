import tensorflow as tf
from pathlib import Path
from skimage.transform import resize
import numpy as np
import h5py
from skimage import io

def Load_Data(path):
    
    total_tf_split = []
    
    for path in Path(path).glob('*.jpg'):
        #Image
        img = tf.keras.utils.load_img(path, color_mode = "rgb", target_size=(256, 256), interpolation='bicubic')
        img = np.asarray(img)/255
        
        #GT_density
        h5_path = str(path).replace('.jpg','.h5').replace('images', 'ground_truth_density')
        f = h5py.File(h5_path, 'r')
        gt = resize(np.asarray(f['density']), (256, 256))
        f.close
        
        #expland dimension
        #img = np.expand_dims(img, -1)
        gt = np.expand_dims(16*gt, -1)
        
        #concat channel 
        tt = np.concatenate((img, gt), axis = -1)
        
        #split and concat tensor
        along_x = tf.split(tt, 1, axis=0)
        along_y = []
        for k in along_x:
            along_y.extend(tf.split(k, 1, axis=1))

        total_split = [tf.expand_dims(i, 0) for i in along_y]
        tf_split = tf.concat(total_split, axis=0)
        
        #Save
        total_tf_split.append(tf_split) 
    
    return tf.concat(total_tf_split, axis=0)


"""
mLoad_Pred:
    Funcion que nos ayuda a crear el tensor de desnidades estimadas del archivo resultante
    luego del entrenamiento.   
"""
def mLoad_Pred(path):
    
    #Cargamos el array usando su ubicacion. Sabemos que esta guardados en un h5file
    f = h5py.File(path, 'r')
    predict_train = tf.constant(np.asarray(f['train']))
    predict_test = tf.constant(np.asarray(f['test']))
    f.close
      
    return predict_train, predict_test


"""
mLoad_GT:
    Funcion que nos ayuda a crear el tensor de imagenes, de canal 1, dada una lista de 
    paths del GT de imagenes. Aqui se reestructura cada array, tanto al insertar un pad
    constante en imagenes con menor shape, y con una dimension adicional que se usa como
    el cana de imagen. El objetivo final es no tener problemas durante la cocatencacion
    de todos estos tensores.  
"""
def mLoad_GT(paths):
    
    #Cargamos el array usando su ubicacion. Sabemos que esta guardados en un h5file
    GT_lts = []
    for path in paths:
        f = h5py.File(path, 'r')
        GT_lts.append(np.asarray(f['density']))
        f.close
    
    #Obtenemos el shape maximo 
    mx, my = max_shape(GT_lts)
    
    
    for i in range(len(GT_lts)):
        #Anadimos el path
        GT = GT_lts[i]
        GT = tf.pad(GT, pad_Tensor([mx, my], GT.shape), "CONSTANT")
        GT_lts[i] = tf.expand_dims(tf.expand_dims(GT, 0), -1)   
        
    
    tt = tf.concat(axis=0)(GT_lts)  
    return tt


"""
mLoad_GT:
    Funcion que nos ayuda a crear el tensor de imagenes, de canal 1, dada una lista de 
    paths de imagenes. Aqui se reestructura cada array, tanto al insertar un pad
    constante en imagenes con menor shape, y con una dimension adicional que se usa como
    el cana de imagen. El objetivo final es no tener problemas durante la cocatencacion
    de todos estos tensores.  
""" 
def mLoad_Img(paths):
    
    #Cargamos el array usando su ubicacion. Sabemos que esta guardados en un h5file
    GT_lts = []
    for path in paths:
        img_path = str(path).replace('.h5','.jpg').replace('ground_truth_density','images').replace('GT_IMG_', 'IMG_')
        img = io.imread(img_path, as_gray = True)
        GT_lts.append(img.astype(np.float64))
    
    #Obtenemos el shape maximo 
    mx, my = max_shape(GT_lts)
    
    
    for i in range(len(GT_lts)):
        GT = GT_lts[i]
        GT = tf.pad(GT, pad_Tensor([mx, my], GT.shape), "CONSTANT")
        GT_lts[i] = tf.expand_dims(tf.expand_dims(GT, 0), -1)
        
    
    tt = tf.concat(axis=0)(GT_lts)  
    return tt
    
"""
max_shape:
    Funcion que nos ayuda a encontrar el shape maximo dada una lista de imagenes.
    Para esto, se crea una lista con la su longitud en cada una de las dimensiones y
    se retorna el maximo alor para cada una de estas listas.
"""
def max_shape(shape_lts):
    shp_x = []
    shp_y = []

    for ff in shape_lts:
        shp_x.append(ff.shape[0])
        shp_y.append(ff.shape[1])
    
    return max(shp_x), max(shp_y)
        

"""
pad_Tensor:
    Funcion que nos ayuda a ajustar el shape de cada imagen en base al shape maximo
    de todas las imagenes. El ajuste se realiza usando un pad constante y que centra 
    la imagen. El objetivo es evitar problemas al concatenar todas las imagenes en 
    un solo tensor.
"""
def pad_Tensor(max_shp, mshape):
    mx, my = max_shp
    
    pd_x = abs(mshape[0] - mx)
    pd_y = abs(mshape[1] - my)
    
    PD_X = []
    PD_Y = []
    tt = []
    
    if (np.mod(pd_x, 2)) == 1:
        pd_x = int(pd_x/2)
        PD_X = [pd_x + 1, pd_x]
    else:
        pd_x = int(pd_x/2)
        PD_X = [pd_x, pd_x]
    
    if (np.mod(pd_y, 2)) == 1:
        pd_y = int(pd_y/2)
        PD_Y = [pd_y + 1, pd_y]
    else:
        pd_y = int(pd_y/2)
        PD_Y = [pd_y, pd_y]
        
    return tf.constant([PD_X, PD_Y])
    
     
        
        
    