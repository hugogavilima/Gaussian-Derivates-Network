import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from scipy.spatial import KDTree 


def gaussian_filter_density(gt, pts):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    f = pts.astype(int)
    leafsize = 2048
    tree = KDTree(f)
    # query kdtree
    #distances, locations = tree.query(pts, k=4)

    for pt in f:
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt] = 1
        distances, locations = tree.query(pt, k=4)
        if gt_count > 1:
            #sigma = (distances[1]+distances[2]+distances[3])*0.1
            sigma = 10
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density



