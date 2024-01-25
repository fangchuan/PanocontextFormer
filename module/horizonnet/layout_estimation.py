import numpy as np
from shapely.geometry import Polygon
import sys
import torch.nn as nn
import os
from module.horizonnet.HorizonNet.model import HorizonNet as BaseHorizon
from module.horizonnet.HorizonNet.misc import post_proc
from module.horizonnet.HorizonNet.inference import find_N_peaks
from module.horizonnet.basic_utils import dict_of_array_to_list_of_dict, list_of_dict_to_dict_of_array
import torch
from module.layout.model.horizon_upsample.upsample1d import Upsample1DHorizonNet
import trimesh
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import SubdivideMeshes,GraphConv
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from module.layout.private_loss import layout_3d_loss
from module.horizonnet.attention import MHSATransformerPos

class HorizonNetNoPostProcess(BaseHorizon):
    def __init__(self, backbone, use_rnn):
        super(HorizonNetNoPostProcess, self).__init__(backbone, use_rnn)

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()

        conv_list = self.feature_extractor(x)
        feature, _ = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor, feature.permute(1,2,0)

class HorizonNetNoPostProcessMesh(BaseHorizon):
    def __init__(self, backbone, use_rnn):
        super(HorizonNetNoPostProcessMesh, self).__init__(backbone, use_rnn)

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()

        conv_list = self.feature_extractor(x)
        feature, feature_list = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor, feature_list


class HorizonNetNoPostProcessUpSample(BaseHorizon):
    def __init__(self, backbone, use_rnn):
        super(HorizonNetNoPostProcessUpSample, self).__init__(backbone, use_rnn)

        self.emb_shared_latent = Upsample1DHorizonNet(1024, 288)

        self.bi_rnn = nn.LSTM(input_size=288,
                              hidden_size=self.rnn_hidden_size,
                              num_layers=2,
                              dropout=0.5,
                              batch_first=False,
                              bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                out_features=3 * 1)
        self.linear.bias.data[0 * 1:1 * 1].fill_(-1)
        self.linear.bias.data[1 * 1:2 * 1].fill_(-0.478)
        self.linear.bias.data[2 * 1:3 * 1].fill_(0.425)

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()

        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)
        feature = self.emb_shared_latent(feature)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, 1)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor, feature.permute(1,2,0)

class HorizonNetUpSample(nn.Module):
    def __init__(self, cfg, optim_spec=None, pretrain_ckpt = None):
        super(HorizonNetUpSample, self).__init__()
        self.cfg = cfg
        self.horizon_net = HorizonNetNoPostProcessUpSample('resnet50', True)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        pretrain_dict = torch.load(pretrain_ckpt, map_location='cpu')
        pretrain_dict = pretrain_dict['model']
        import collections
        new_pretrain_dict = collections.OrderedDict()
        len_key = len('horizon_net.module.')
        len_key2 = len('layout_estimation_net.horizon_net.module.')
        for key in pretrain_dict:
            if (key[:len_key] == 'horizon_net.module.'):
                new_pretrain_dict[key[len_key:]] = pretrain_dict[key]
            elif (key[:len_key2]=='layout_estimation_net.horizon_net.module.'):
                new_pretrain_dict[key[len_key2:]] = pretrain_dict[key]
        self.horizon_net.load_state_dict(new_pretrain_dict, strict=True)

        self.horizon_net = nn.DataParallel(self.horizon_net)


    def forward(self, x):
        bon, cor, layout_feature = self.horizon_net(x)
        horizon_layout = {'initial_bon': bon, 'initial_cor': cor, 'layout_feature': layout_feature}

        return horizon_layout

class HorizonNet(nn.Module):
    def __init__(self, cfg, optim_spec=None, pretrain_ckpt = None):
        super(HorizonNet, self).__init__()
        self.cfg = cfg
        self.horizon_net = HorizonNetNoPostProcess('resnet50', True)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        if(pretrain_ckpt != None):
            if(os.path.exists(pretrain_ckpt)):
                print("loading ",pretrain_ckpt)
                try:
                    pretrain_dict = torch.load(pretrain_ckpt, map_location='cpu')['state_dict']
                    self.horizon_net.load_state_dict(pretrain_dict, strict=True)
                except:
                    pretrain_dict = torch.load(pretrain_ckpt, map_location='cpu')
                    import collections
                    new_pretrain_dict = collections.OrderedDict()
                    if('model' in pretrain_dict):
                        pretrain_dict = pretrain_dict['model']
                        len_key = len('horizon_net.module.')
                    if('net' in pretrain_dict):
                        pretrain_dict = pretrain_dict['net']
                        len_key = len('layout_estimation.horizon_net.module.')
                        len_key2 = len('layout_estimation.module.')
                    for key in pretrain_dict:
                        if (key[:len_key] == 'horizon_net.module.'):
                            new_pretrain_dict[key[len_key:]] = pretrain_dict[key]
                        elif (key[:len_key] == 'layout_estimation.horizon_net.module.'):
                            new_pretrain_dict[key[len_key:]] = pretrain_dict[key]
                        elif (key[:len_key2] == 'layout_estimation.module.'):
                            new_pretrain_dict[key[len_key2:]] = pretrain_dict[key]

                    self.horizon_net.load_state_dict(new_pretrain_dict,strict=True)
            else:
                print(pretrain_ckpt," is not exist!")
        self.horizon_net = nn.DataParallel(self.horizon_net)

    def forward(self, x):
        bon, cor, layout_feature = self.horizon_net(x)
        horizon_layout = {'initial_bon': bon, 'initial_cor': cor, 'layout_feature': layout_feature}

        return horizon_layout

        if self.training:
            return horizon_layout
        else:
            height, width = x.shape[2:]
            # transform pixel layout estimation to pixel manhattan world layout
            manhattan_pix = []
            layout_scenes = dict_of_array_to_list_of_dict(horizon_layout)
            for layout_scene in layout_scenes:
                try:
                    dt_cor_id, z0, z1 = horizon_to_manhattan_layout(
                        layout_scene, height, width, force_cuboid=False)
                except:
                    dt_cor_id = np.array([
                        [k // 2 * 1024, 256 - ((k % 2) * 2 - 1) * 120]
                        for k in range(8)
                    ])
                manhattan_pix.append(dt_cor_id)
            return horizon_layout, manhattan_pix

class HorizonNetMesh(nn.Module):
    def __init__(self, cfg, optim_spec=None, pretrain_ckpt = None):
        super(HorizonNetMesh, self).__init__()
        self.cfg = cfg
        self.horizon_net = HorizonNetNoPostProcessMesh('resnet50', True)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        if(pretrain_ckpt != None):
            if(os.path.exists(pretrain_ckpt)):
                print("loading ",pretrain_ckpt)
                try:
                    pretrain_dict = torch.load(pretrain_ckpt, map_location='cpu')['state_dict']
                    self.horizon_net.load_state_dict(pretrain_dict, strict=True)
                except:
                    pretrain_dict = torch.load(pretrain_ckpt, map_location='cpu')
                    import collections
                    new_pretrain_dict = collections.OrderedDict()
                    if('model' in pretrain_dict):
                        pretrain_dict = pretrain_dict['model']
                        len_key = len('horizon_net.module.')
                    if('net' in pretrain_dict):
                        pretrain_dict = pretrain_dict['net']
                        len_key = len('layout_estimation.horizon_net.module.')
                        len_key2 = len('layout_estimation.module.')
                    for key in pretrain_dict:
                        if (key[:len_key] == 'horizon_net.module.'):
                            new_pretrain_dict[key[len_key:]] = pretrain_dict[key]
                        elif (key[:len_key] == 'layout_estimation.horizon_net.module.'):
                            new_pretrain_dict[key[len_key:]] = pretrain_dict[key]
                        elif (key[:len_key2] == 'layout_estimation.module.'):
                            new_pretrain_dict[key[len_key2:]] = pretrain_dict[key]

                    self.horizon_net.load_state_dict(new_pretrain_dict,strict=True)
            else:
                print(pretrain_ckpt," is not exist!")

        self.loss_3d = layout_3d_loss.layout_3dloss()

        self.sample_gap = 8

        self.mhsa = MHSATransformerPos(num_layers=1, d_model=480, num_heads=4, conv_hidden_dim=2048,
                                       maximum_position_encoding=(8192//self.sample_gap-6))
        self.drop_out = nn.Dropout(0.5)

        self.reduce_feats = True
        if (self.reduce_feats):
            self.bottleneck = nn.Linear(480, 128)
            nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.bottleneck.bias, 0)

        self.gconvs = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        for i in range(6):
            input_dim = 128 + 3
            gconv = GraphConv(input_dim, 128, init="normal", directed=False)
            self.gconvs.append(gconv)

        self.vert_offset = nn.Linear(128 + 3, 3, bias=False)

    def slice_projection(self, uv_inputs, img_feature):

        uv_inputs = uv_inputs.to(img_feature.device)
        uv_inputs = uv_inputs.unsqueeze(1)

        output = F.grid_sample(img_feature, uv_inputs, align_corners=True)
        output = torch.transpose(output.squeeze(2), 1, 2)

        return output

    def _padded_to_packed(self, x, idx):
        D = x.shape[-1]
        idx = idx.view(-1, 1).expand(-1, D)

        x_packed = x.view(-1, D).gather(0, idx.to(x.device))

        return x_packed

    def xyz2uv(self, xyz, eps=0.001):
        x, y, z = torch.unbind(xyz, dim=2)

        x = -x + eps
        y = -y + eps
        z = z + eps

        u = torch.atan2(x, -y)
        v = - torch.atan(z / torch.sqrt(
            x ** 2 + y ** 2))  ###  (default: - for z neg (under horizon) - grid sample instead expects -1,-1 top-left

        pi = float(np.pi)

        u = u / pi
        v = (2.0 * v) / pi

        u = torch.clamp(u, min=-1, max=1)
        v = torch.clamp(v, min=-1, max=1)

        ###output: [batch_size x num_points x 2]##range -1,+1

        output = torch.stack([u, v], dim=-1)

        return output

    def forward(self, inputs):
        bon, cor, layout_feature = self.horizon_net(inputs['image'])
        horizon_layout = {'initial_bon': bon, 'initial_cor': cor, 'layout_feature': layout_feature}

        self.loss_3d.setGrid(inputs['unit_xyz'][0, ...][None, None, ...])
        pred_lonlat_down = torch.cat([inputs['unit_lonlat'][:, :, 0:1], horizon_layout['initial_bon'][:, 1, :, None]], dim=-1)
        layout_down_pose, _ = self.loss_3d.lonlat2xyz_down(pred_lonlat_down)

        layout_down_pose = layout_down_pose.detach().clone()
        layout_down_pose_trans = torch.cat([-layout_down_pose[:, :, 0].unsqueeze(2), layout_down_pose[:, :, 2].unsqueeze(2), -layout_down_pose[:, :, 1].unsqueeze(2)], dim=2)
        layout_down_pose_trans_sample = layout_down_pose_trans.detach().clone()
        bs = layout_down_pose_trans_sample.shape[0]
        gt_verts_list = []
        gt_faces_list = []
        for bs_idex in range(bs):
            layout_down_pose_idx = layout_down_pose_trans_sample[bs_idex, 0:1024:self.sample_gap, :2]
            polygon = Polygon(layout_down_pose_idx.tolist())
            if(not polygon.is_valid):
                polygon = polygon.buffer(0)
            transformation = np.eye(4)
            transformation[2, 3] = -1.6
            mesh = trimesh.creation.extrude_polygon(polygon, height=2.4, transform=transformation)
            gt_verts_list.append(torch.tensor(mesh.vertices, dtype=torch.float32).to(layout_down_pose_trans_sample.device))
            gt_faces_list.append(torch.tensor(mesh.faces).to(layout_down_pose_trans_sample.device))

        pytorch_mesh = Meshes(verts=gt_verts_list, faces=gt_faces_list)

        subdivide = SubdivideMeshes()
        mesh_subdivide = subdivide(pytorch_mesh)
        # save_obj('subdivide.obj',mesh_subdivide.verts_packed(), mesh_subdivide.faces_packed())

        verts_padded_to_packed_idx = mesh_subdivide.verts_padded_to_packed_idx()
        vert_pos_padded = mesh_subdivide.verts_padded()
        vert_pos_packed = self._padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)

        uv_inputs = self.xyz2uv(vert_pos_padded)
        feats = []

        for img_feature in horizon_layout['layout_feature']:
            feats.append( self.slice_projection(uv_inputs, img_feature))

        output = torch.cat(feats, 2)
        output = self.mhsa(output)
        output = self.drop_out(output)

        vert_align_feats = self._padded_to_packed(output, verts_padded_to_packed_idx)

        if (self.reduce_feats):
            vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        first_layer_feats = [vert_align_feats, vert_pos_packed]

        vert_feats = torch.cat(first_layer_feats, dim=1)

        ep = mesh_subdivide.edges_packed()

        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, ep))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

        # Predict a new mesh by offsetting verts
        vert_offsets = torch.tanh(self.vert_offset(vert_feats))

        mesh_subdivide_update = mesh_subdivide.offset_verts(vert_offsets)

        horizon_layout['pred_mesh'] = mesh_subdivide_update
        # save_obj('subdivide_update.obj', mesh_subdivide_update.verts_packed(), mesh_subdivide_update.faces_packed())

        return horizon_layout


def horizon_to_manhattan_layout(horizon_layout, H, W, force_cuboid=True, min_v=None, r=0.05):
    y_bon_, y_cor_  = horizon_layout['bon'], horizon_layout['cor']

    y_bon_ = (y_bon_ / np.pi + 0.5) * H - 0.5
    y_cor_ = y_cor_[0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j * 2] = cor[j, 0], cor[j, 1]
        cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]

    # # Normalized to [0, 1]
    # cor_id[:, 0] /= W
    # cor_id[:, 1] /= H

    return cor_id, z0, z1
