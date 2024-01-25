import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.layout.private_loss.EquirecCoordinate import EquirecTransformer

class layout_3dloss(nn.Module):
    def __init__(self, camera_height=1.6):
        super(layout_3dloss, self).__init__()
        assert camera_height > 0
        self.cH = camera_height
        self.grid = None
        self.c2d = Corner2Depth(None)
        self.et = EquirecTransformer('torch')

    def setGrid(self, grid):
        self.grid = grid
        self.c2d.setGrid(grid)

    def lonlat2xyz_up(self, pred_up, GT_up, up_down_ratio):
        pred_up_xyz = self.et.lonlat2xyz(pred_up)
        GT_up_xyz = self.et.lonlat2xyz(GT_up)

        s = -(self.cH * up_down_ratio[..., None, None]) / pred_up_xyz[..., 1:2].detach()
        pred_up_xyz *= s

        s = -(self.cH * up_down_ratio[..., None, None]) / GT_up_xyz[..., 1:2]
        GT_up_xyz *= s

        return pred_up_xyz, GT_up_xyz

    def lonlat2xyz_down(self, pred_down):
        pred_down_xyz = self.et.lonlat2xyz(pred_down)
        s = self.cH / pred_down_xyz[..., 1:2].detach()
        pred_down_xyz *= s

        return pred_down_xyz, None

    def cal_uniform_weight(self,GT_up_xyz):
        GT_up_xyz_roll = torch.roll(GT_up_xyz, 1, 1)
        diff = GT_up_xyz_roll - GT_up_xyz
        distance = torch.norm(diff, p=2, dim=2)
        # weight = (next_dis/torch.mean(distance,1).unsqueeze(1)).unsqueeze(2)
        weight = distance-torch.mean(distance,1).unsqueeze(1)
        ones = torch.ones(weight.shape).to(GT_up_xyz.device)
        twos = torch.ones(weight.shape).to(GT_up_xyz.device)*2
        weight = torch.where(weight>0,twos,ones).unsqueeze(2)
        weight_return = torch.cat((weight,weight,weight),2)

        return weight_return.to(GT_up_xyz.device)

    def forward(self, pred_up, pred_down, GT_up, up_down_ratio, GT_up_xz, GT_down_xz, sample_weight=False):

        assert self.grid is not None
        # _, GT_up_xyz_cor = self.lonlat2xyz_up(pred_up, GT_up, up_down_ratio)
        pred_up_xyz, GT_up_xyz = self.lonlat2xyz_up(pred_up, GT_up_xz, up_down_ratio)
        if sample_weight:
            weight = self.cal_uniform_weight(GT_up_xyz)
            print("not using sample_weight")
            exit()
        else:
            weight = torch.ones((GT_up_xyz.shape))

        pred_down_xyz, _ = self.lonlat2xyz_down(pred_down)
        GT_down_xyz, _ = self.lonlat2xyz_down(GT_down_xz)

        # self.vis_result(GT_up_xyz_cor[0,:], GT_up_xyz[0,:], pred_up_xyz[0,:], pred_down_xyz[0,:], GT_down_xyz[0,:], weight[0,:])
        if sample_weight:
            loss_3d_up_down = F.l1_loss(pred_up_xyz*weight, GT_up_xyz*weight) + F.l1_loss(pred_down_xyz*weight, GT_down_xyz*weight)
        else:
            loss_3d_up_down = F.l1_loss(pred_up_xyz, GT_up_xyz) + F.l1_loss(pred_down_xyz, GT_down_xyz)

        return loss_3d_up_down, pred_up_xyz, pred_down_xyz

    # def vis_result(self,GT_up_xyz_cor, GT_up_xyz, pred_up_xyz, pred_down_xyz, GT_down_xyz,weight):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     GT_up_xyz_cor = np.array(GT_up_xyz_cor).squeeze()
    #     GT_up_xyz = np.array(GT_up_xyz).squeeze()
    #     GT_down_xyz = np.array(GT_down_xyz).squeeze()
    #     pred_up_xyz = np.array(pred_up_xyz.detach()).squeeze()
    #     pred_down_xyz = np.array(pred_down_xyz.detach()).squeeze()
    #     for i in range(1024):
    #         if (weight[i,0]<1.1):
    #             continue
    #         plt.plot(GT_up_xyz[i,0] , GT_up_xyz[i,2], '.', markersize=5., color='r')
    #     for i in range(4):
    #         plt.plot(GT_up_xyz_cor[i, 0], GT_up_xyz_cor[i, 2], 'o', markersize=5., color='b')
    #     plt.plot( 0, 0, 'o', markersize=10., color='k')
    #     for i in range(1024):
    #         plt.plot(pred_up_xyz[i,0] , pred_up_xyz[i,2], '.', markersize=5., color='g')
    #     for i in range(1024):
    #         plt.plot(pred_down_xyz[i,0] , pred_down_xyz[i,2], '.', markersize=5., color='y')
    #     plt.axis('equal')
    #     plt.show()

class Corner2Depth(nn.Module):
    def __init__(self, grid):
        super(Corner2Depth, self).__init__()
        self.grid = grid

    def setGrid(self, grid):
        self.grid = grid

    def forward(self, corners, nums, shift=None, mode='origin'):
        if mode == 'origin':
            return self.forward_origin(corners, nums, shift)
        else:
            return self.forward_fast(corners, nums, shift)

    def forward_fast(self, corners, nums, shift=None):
        if shift is not None: raise NotImplementedError
        grid_origin = self.grid.to(corners.device)
        eps = 1e-2
        depth_maps = []
        normal_maps = []

        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, ...].clone()
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            diff = corners_now[1:, ...] - corners_now[:-1, ...]
            vec_yaxis = torch.zeros_like(diff)
            vec_yaxis[..., 1] = 1
            cross_result = torch.cross(diff, vec_yaxis, dim=1)
            d = -torch.sum(cross_result * corners_now[:-1, ...], dim=1, keepdim=True)
            planes = torch.cat([cross_result, d], dim=1)
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)

            intersec = []
            for idx in range(scale_all.shape[-1]):
                intersec.append((grid * scale_all[..., idx:idx + 1]).unsqueeze(-1))
            intersec = torch.cat(intersec, dim=-1)
            a = corners_now[1:, ...]
            b = corners_now[:-1, ...]

            x_cat = torch.cat([a[:, 0:1], b[:, 0:1]], dim=1)
            z_cat = torch.cat([a[:, 2:], b[:, 2:]], dim=1)

            max_x, min_x = torch.max(x_cat, dim=1)[0], torch.min(x_cat, dim=1)[0]
            max_z, min_z = torch.max(z_cat, dim=1)[0], torch.min(z_cat, dim=1)[0]

            mask_x = (intersec[:, :, :, 0, :] <= max_x + eps) & (intersec[:, :, :, 0, :] >= min_x - eps)
            mask_z = (intersec[:, :, :, 2, :] <= max_z + eps) & (intersec[:, :, :, 2, :] >= min_z - eps)
            mask_valid = scale_all > 0
            mask = ~(mask_x & mask_z & mask_valid)
            # scale_all[mask] = float('inf')
            scale_all[mask] = float(3e2)

            depth, min_idx = torch.min(scale_all, dim=-1)
            _, h, w = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(1, h, w, -1)

            depth_maps.append(depth)
            normal_maps.append(normal)
        depth_maps = torch.cat(depth_maps, dim=0).unsqueeze(1)
        normal_maps = torch.cat(normal_maps, dim=0)

        return depth_maps, normal_maps

    def forward_origin(self, corners, nums, shift=None):
        # corners is (bs, 12, 3)
        # nums is (bs, )
        # shift is bs x 2 which are x and z shift
        grid_origin = self.grid.to(corners.device)
        eps = 1e-2
        depth_maps = []
        normal_maps = []
        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, :num, ...].clone()  # num x 3
            if shift is not None:
                corners_now[..., 0] -= shift[i, 0]
                corners_now[..., 2] -= shift[i, 1]
            # equation: ax + by + cz + d = 0
            #
            corners_now = torch.cat(
                [corners_now, corners_now[0:1, ...]], dim=0)
            planes = []
            for j in range(1, corners_now.shape[0]):
                vec_corner = corners_now[j:j + 1, ...] - corners_now[j - 1:j, ...]
                vec_yaxis = torch.zeros_like(vec_corner)
                vec_yaxis[..., 1] = 1
                cross_result = torch.cross(vec_corner, vec_yaxis)
                # now corss_result is a b c
                cross_result = cross_result / \
                               torch.norm(cross_result, p=2, dim=-1)[..., None]
                d = -torch.sum(cross_result *
                               corners_now[j:j + 1, ...], dim=-1)[..., None]
                abcd = torch.cat([cross_result, d], dim=-1)  # abcd is 1, 4
                planes.append(abcd)
            planes = torch.cat(planes, dim=0)  # planes is num x 4
            assert planes.shape[0] == num
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)
            depth = []
            for j in range(scale_all.shape[-1]):
                scale = scale_all[..., j]
                intersec = scale[..., None] * grid
                a = corners_now[j + 1:j + 2, :]
                b = corners_now[j:j + 1, :]
                rang = torch.cat([a, b], dim=0)
                max_x, min_x = torch.max(rang[:, 0]), torch.min(rang[:, 0])
                max_z, min_z = torch.max(rang[:, 2]), torch.min(rang[:, 2])

                mask_x = (intersec[..., 0] <= max_x +
                          eps) & (intersec[..., 0] >= min_x - eps)
                mask_z = (intersec[..., 2] <= max_z +
                          eps) & (intersec[..., 2] >= min_z - eps)
                mask_valid = scale > 0
                mask = ~ (mask_x & mask_z & mask_valid)

                scale[mask] = float(3e2)
                depth.append(scale[None, ...])

            depth = torch.cat(depth, dim=1)
            depth, min_idx = torch.min(depth, dim=1)
            [_, h, w] = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(-1, h, w, 3)
            normal_maps.append(normal)
            depth_maps.append(depth[None, ...])
        depth_maps = torch.cat(depth_maps, dim=0)
        normal_maps = torch.cat(normal_maps, dim=0)
        return depth_maps, normal_maps