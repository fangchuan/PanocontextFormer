import numpy as np
import os
import torch
from module.layout.dataloader.layout_dataloder_utils import cor_2_1d
from module.layout.private_loss import XY2xyz,lonlat2xyz
from module.layout.misc.post_proc import np_coor2xy,np_coory2v
import trimesh

class LayoutDataset(object):

    def __init__(self):
        _, self.unit_lonlat, self.unit_xyz = self.create_grid([512, 1024])

    def create_grid(self, shape):
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

    def save_wirefame(self, cor):
        W = 1024
        H = 512
        N = len(cor) // 2
        floor_z = -1.6
        floor_xy = np_coor2xy(cor[1::2], floor_z, W, H, floorW=1, floorH=1)
        c = np.sqrt((floor_xy**2).sum(1))
        v = np_coory2v(cor[0::2, 1], H)
        ceil_z = (c * np.tan(v)).mean()

        wf_points = [[-x, -y, floor_z] for x, y in floor_xy] +\
                    [[-x, -y, ceil_z] for x, y in floor_xy]
        wf_points = np.array(wf_points)
        wf_lines = [[i, (i+1)%N] for i in range(N)] +\
                   [[i+N, (i+1)%N+N] for i in range(N)] +\
                   [[i, i+N] for i in range(N)]
        mesh = []
        for indexes in wf_lines:
            line = wf_points[indexes]
            line_mesh = trimesh.creation.cylinder(0.025, sections=8, segment=line)
            mesh.append(line_mesh)
        mesh = sum(mesh)
        trimesh.exchange.export.export_mesh(mesh, os.path.join('layout_mesh.ply'))

    def get_3dloss_info(self,cor):

        coor_u_up = cor[::2, 0]
        coor_v_up = cor[::2, 1]
        coor_v_down = cor[1::2, 1]

        # coor2rad
        coor_u_up = (coor_u_up.copy() /1024 - 0.5) * 2 * np.pi
        coor_v_up = (coor_v_up.copy() / 512 - 0.5) * np.pi
        coor_v_down = (coor_v_down.copy() / 512 - 0.5) * np.pi

        c0 = 1.6 / np.tan(coor_v_down)
        z1 = c0 * np.tan(coor_v_up)
        z1_mean = np.mean(z1)
        ceiling_floor_ratio = np.abs(z1_mean/1.6)

        new_pts = np.zeros([100, 2], np.float32) + 10000 # option
        new_pts[:coor_u_up.shape[0], 0] = coor_u_up
        new_pts[:coor_u_up.shape[0], 1] = coor_v_up

        return coor_u_up.shape[0], ceiling_floor_ratio, new_pts

    def get_layout_data(self, cor):
        H, W = 512, 1024
        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, H, W)
        # Prepare lrub and occ
        corx = np.round(cor[::2, 0])
        xs = []
        lrub = []
        occ = []
        for i in range(len(corx)):
            u, b = cor[[i * 2, i * 2 + 1], 1]
            if corx[i] == corx[i - 1]:
                lrub[-1][2] = u
                lrub[-1][3] = b
                occ[-1] = 1
            else:
                xs.append(cor[i * 2, 0])
                lrub.append([u, b, u, b])
                occ.append(0)
        xs = np.array(xs)
        lrub = np.array(lrub)
        occ = np.array(occ)
        uidx = np.arange(W)
        cdist = np.abs(uidx[:, None] - xs[None, :])
        cdist[cdist > W / 2] = W - cdist[cdist > W / 2]
        nnidx = cdist.argmin(1)
        lrub = lrub[nnidx].T
        lrub = ((lrub + 0.5) / H - 0.5) * np.pi
        occ = occ[nnidx][None]
        wall_num, ceiling_floor_ratio, gt_up_uv = self.get_3dloss_info(cor)

        # Prepare 1d wall-wall probability
        uidx = np.arange(W)
        corx = np.round(np.unique(cor[:, 0])).astype(np.int32)
        corx = np.concatenate([corx, corx + W, corx - W])
        dist = corx[:, None] - uidx[None, :]
        vot = dist[None, np.abs(dist).argmin(0), uidx]

        # Convert all data to tensor
        out_dict = {
            'bon': torch.FloatTensor(bon.copy()),
            'vot': torch.FloatTensor(vot.copy()),
            'cor': 0.96 ** torch.FloatTensor(vot.copy()).abs(),
            'lrub': torch.FloatTensor(lrub.copy()),
            'occ': torch.FloatTensor(occ.copy()),
            'gt_up_uv': gt_up_uv,
            'ratio': ceiling_floor_ratio.astype(np.float32),
            'unit_lonlat': self.unit_lonlat,
            'unit_xyz': self.unit_xyz,
        }

        return out_dict