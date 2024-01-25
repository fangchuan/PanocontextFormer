import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.layout.model import backbone
from module.layout.model import horizon_compression
from module.layout.model import horizon_refinement
from module.layout.model import horizon_upsample
from module.layout.model import modality
from module.layout.model.utils import wrap_lr_pad
from module.layout.model.backbone import Resnet
from module.layout.model.horizon_compression.ehc import EfficientHeightReduction
from module.layout.model.horizon_refinement.attention import TransEn
from module.layout.model.horizon_upsample.upsample1d import Upsample1D

import math

from module.layout.config.config import config, update_config,infer_exp_id
import argparse
'''
HoHoNet
'''

class HoHoNet(nn.Module):
    def __init__(self, emb_dim=256, input_hw=None, input_norm='imagenet', pretrain='',
                 backbone_config={'module': 'Resnet'},
                 decode_config={'module': 'EfficientHeightReduction'},
                 refine_config={'module': 'TransEn'},
                 upsample_config={'module': 'Upsample1D'},
                 modalities_config={}):
        super(HoHoNet, self).__init__()

        self.input_hw = input_hw
        self.bon_scale = 1.
        self.post_force_cuboid = False
        emb_dim = 288
        if input_norm == 'imagenet':
            self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
            self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))
        else:
            raise NotImplementedError

        # Encoder
        self.encoder = Resnet(backbone = 'resnet34')

        # Horizon compression convert backbone features to horizontal feature
        # I name the variable as decoder during development and forgot to fix :P
        self.decoder = EfficientHeightReduction(self.encoder.out_channels, self.encoder.feat_heights)

        # Horizontal feature refinement module
        self.horizon_refine = TransEn(c_mid = self.decoder.out_channels, position_encode = 256, nhead = 8, num_layers = 1, dim_feedforward= 2048)

        # Channel reduction to the shared latent
        self.emb_shared_latent = Upsample1D(self.horizon_refine.out_channels, emb_dim)

        # Instantiate desired modalities
        # config
        oneconv = True
        last_ks = 1
        last_bias = False
        dropout = 0
        if oneconv:
            self.pred_bon = nn.Conv1d(emb_dim, 2, last_ks, padding=last_ks // 2, bias=last_bias)
            self.pred_cor = nn.Conv1d(emb_dim, 1, last_ks, padding=last_ks // 2, bias=last_bias)
            if last_bias:
                nn.init.constant_(self.pred_bon.bias[0], -0.478)
                nn.init.constant_(self.pred_bon.bias[1], 0.425)
                nn.init.constant_(self.pred_cor.bias, -1.)
        else:
            self.pred_bon = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, 3, padding=1, bias=False),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(emb_dim, 2, 1),
            )
            self.pred_cor = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, 3, padding=1, bias=False),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(emb_dim, 1, 1),
            )
            nn.init.constant_(self.pred_bon[-1].bias[0], -0.478)
            nn.init.constant_(self.pred_bon[-1].bias[1], 0.425)
            nn.init.constant_(self.pred_cor[-1].bias, -1.)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        # Patch for all conv1d/2d layer's left-right padding
        wrap_lr_pad(self)

        # Load pretrained
        if pretrain:
            print(f'Load pretrained {pretrain}')
            st = torch.load(pretrain, map_location='cpu')
            st = st['model']
            import collections
            new_st = collections.OrderedDict()
            for key in st:
                if(key[29:] in self.state_dict()):
                    new_st[key[29:]] = st[key]
            st = new_st
            missing_key = self.state_dict().keys() - st.keys()
            unknown_key = st.keys() - self.state_dict().keys()
            print('Missing key:', missing_key)
            print('Unknown key:', unknown_key)
            self.load_state_dict(st, strict=True)

    def extract_feat(self, x):
        ''' Map the input RGB to the shared latent (by all modalities) '''

        if self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode='bilinear', align_corners=False)
        x = (x - self.x_mean) / self.x_std
        # encoder
        # b*64*128*256/b*128*64*128/b*256*32*64/b*512*16*32
        conv_list = self.encoder(x)
        # decoder to get horizontal feature
        # feat_1D_size: 1024*256*1
        feat = self.decoder(conv_list)

        # refine feat [1*1024*256]
        feat = self.horizon_refine(feat)
        # embed the shared latent feature_1D [1,256,1024]
        feat = self.emb_shared_latent(feat)
        return feat

    def forward(self, inputs, end_points = None):

        if end_points==None:
            end_points = {}

        # extract 1D-featrue
        feat = self.extract_feat(inputs['image'])
        end_points['layout_featrue'] = feat

        # recover bon cor
        x_emb = feat['1D']
        if self.dropout is not None:
            x_emb = self.dropout(x_emb)
        pred_bon = self.pred_bon(x_emb)
        pred_cor = self.pred_cor(x_emb)

        self.sigmod_normlize = True
        if self.sigmod_normlize:
            bon = torch.nn.functional.sigmoid(pred_bon)
            up = bon[:, 0:1, :] * -0.5 * torch.pi
            down = bon[:, 1:, :] * 0.5 * torch.pi
            pred_bon = torch.cat([up, down], dim=1)
        end_points['pred_bon'] = pred_bon
        end_points['pred_cor'] = pred_cor

        return end_points

    def infer(self, inputs, end_points = None):
        from module.layout.misc.post_proc import np_refine_by_fix_z, gen_ww, infer_coory
        from scipy.ndimage.filters import maximum_filter
        from shapely.geometry import Polygon

        self.eval()
        with torch.no_grad():
            end_points = self(inputs)
        pred_bon = end_points['pred_bon'].clone()
        pred_cor = end_points['pred_cor'].clone()
        pred_bon = pred_bon / self.bon_scale
        H, W = 512, 1024

        y_bon_ = (pred_bon[0].cpu().numpy() / np.pi + 0.5) * H - 0.5
        y_cor_ = pred_cor[0, 0].sigmoid().cpu().numpy()
        # Init floor/ceil plane
        z0 = 50
        _, z1 = np_refine_by_fix_z(*y_bon_, z0)

        # Detech wall-wall peaks
        def find_N_peaks(signal, r, min_v, N):
            max_v = maximum_filter(signal, size=r, mode='wrap')
            pk_loc = np.where(max_v == signal)[0]
            pk_loc = pk_loc[signal[pk_loc] > min_v]
            if N is not None:
                order = np.argsort(-signal[pk_loc])
                pk_loc = pk_loc[order[:N]]
                pk_loc = pk_loc[np.argsort(pk_loc)]
            return pk_loc, signal[pk_loc]

        min_v = 0 if self.post_force_cuboid else 0.05
        r = int(round(W * 0.05 / 2))
        N = 4 if self.post_force_cuboid else None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        # Generate wall-walls
        cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6),
                                       force_cuboid=self.post_force_cuboid)
        if not self.post_force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
            if not Polygon(xy2d).is_valid:
                import sys
                print(
                    'Fail to generate valid general layout!! '
                    'Generate cuboid as fallback.',
                    file=sys.stderr)
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

        # Expand with btn coory
        cor = np.hstack([cor, infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor) * 2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j * 2] = cor[j, 0], cor[j, 1]
            cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]
        return {'cor_id': cor_id, 'y_bon_': y_bon_, 'y_cor_': y_cor_}



if __name__ == '__main__':


    net = HoHoNet()
    inputs = {}
    inputs['image'] = torch.zeros((1,3,512,1024))
    output_dict = net(inputs)