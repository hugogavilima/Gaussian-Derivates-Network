import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from scipy.spatial import KDTree
from progressbar import progressbar 
from scipy import io 
import skimage
import h5py
from pathlib import Path


"""
gaussian_filter_density:
    Funcion que recibe un array binario del shape de nuestra imagen, con un valor de verdadero donde 
    se encuentre una cabeza. Utilizamos geometric adapatative kernels para estimar la distorcion de perspectiva 
    en escenarios muy aglomerados.
    La funcion devuelve una rray del mismo shape con el mapa de densidad estimado. Ademas de un array unidimensional
    con los paramateros sigmas utilizados. Este array tiene longitud igual a la cantidad de cabezas en la imagen.
"""

def gaussian_filter_density(gts):
    density = np.zeros(gts.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gts)
    if gt_count == 0:
        return density

    
    leafsize = 2048
    gts_points =  np.transpose(gts.nonzero())
    #tree = KDTree(gts_points, leafsize= leafsize)

    result = np.array([], dtype = np.double)
    
    for pt in gts_points:
        pt2d = np.zeros(gts.shape, dtype=np.float32)
        try:
            pt2d[pt[0], pt[1]] = 1
        except:
            print(pt)

        #distances, locations = tree.query(pt, k=5)
        if gt_count > 1:
           #sigma = (distances[1] + distances[2] + distances[3])*0.1
           sigma = 15
        else:
           #sigma = np.average(np.array(gts.shape))/2./2. #case: 1 point
           sigma = 15

        density += gaussian_filter(pt2d, sigma, mode='constant')
        np.append(result, sigma)
        
    return density, result


"""
GT_generation:
    Funcion que recibe como entrada una lista de rutas de imagenes, y generar los mapa de 
    densidad respectivos a cada imagen en la lista, y los alamcena en un archivo h5py

    La funcion devuelve una lista con los paths de los archivos h5py resultantes.
"""
def GT_generation(img_paths):
    progress = progressbar.ProgressBar()
    path_names = []
    for img_path in progress(img_paths):

        mat = io.loadmat(str(img_path).replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img = skimage.io.imread(img_path, as_gray = True)

        #aqui solo tomamos el alto y el ancho de la imagen
        k = np.zeros((img.shape[0],img.shape[1]))

        #Hacemos esto porque la wea de .mat se guarda raro
        gt = mat["image_info"][0,0][0,0][0]
        
        #Validamos que esten dentro de nuestra matriz, y colocamos un uno donde encontramos una cabeza
        for i in range(0,len(gt)):
            if (int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]):
                #rr, cc = disk((gt[i][1], gt[i][0]), 10, shape=k.shape)
                k[int(gt[i][1]), int(gt[i][0])] = 1


        fg, distance = gaussian_filter_density(k)      
        name = str(img_path).replace('.jpg','.h5').replace('images','ground_truth_density')
        with h5py.File(name, 'w') as hf:
            hf['density'] = fg
            hf['distance'] = distance

        path_names.append(name)
    
    return path_names


def ls_paths(PTH):
    img_paths = []
    for path in Path(PTH).glob('*.jpg'):
        img_paths.append(path)
    return img_paths

 