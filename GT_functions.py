import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from scipy.spatial import KDTree 


def gaussian_filter_density(gts):
    density = np.zeros(gts.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gts)
    if gt_count == 0:
        return density

    
    leafsize = 2048
    gts_points =  np.transpose(gts.nonzero())
    tree = KDTree(gts_points, leafsize= leafsize)

    result = np.array([], dtype = np.double)
    
    for pt in gts_points:
        pt2d = np.zeros(gts.shape, dtype=np.float32)
        try:
            pt2d[pt[0], pt[1]] = 1
        except:
            print(pt)

        distances, locations = tree.query(pt, k=5)
        if gt_count > 1:
           sigma = (distances.mean())*0.3
           #sigma = 1
        else:
           sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

        density += gaussian_filter(pt2d, sigma, mode='constant')
        np.append(result, sigma)
        
    return density, result


