
import numpy as np
from module.layout.private_loss.EquirecCoordinate import XY2xyz,lonlat2xyz

def create_grid(shape):
    h, w = shape
    X = np.tile(np.arange(w)[None, :, None], (h, 1, 1))
    Y = np.tile(np.arange(h)[:, None, None], (1, w, 1))
    XY = np.concatenate([X, Y], axis=-1)
    xyz = XY2xyz(XY, shape, mode='numpy')
    l = w
    mean_lonlat = np.zeros([l, 2], dtype=np.float32)
    mean_lonlat[:, 1] = 0
    mean_lonlat[:, 0] = ((np.arange(l) / float(l - 1)) * 2 * np.pi - np.pi).astype(np.float32)
    mean_xyz = lonlat2xyz(mean_lonlat, mode='numpy')

    return xyz, mean_lonlat, mean_xyz