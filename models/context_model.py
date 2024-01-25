import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from .backbone_module import Pointnet2Backbone
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead, \
    ClsAgnosticPredictHead
from pytorch3d.ops import GraphConv


class ContextModule(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 size_cls_agnostic=False, emb_codes_dim = 0, image_feature_fusion = False, layout_flag = False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic
        self.emb_codes_dim = emb_codes_dim

        self.decoder_obj_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_vert_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder = nn.ModuleList()
        self.decoder.append(TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(3, 288),
                    cross_posembed=PositionEmbeddingLearned(6, 288),
                ))

        self.decoder.append(TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(6, 288),
                    cross_posembed=PositionEmbeddingLearned(3, 288),
                ))

        self.use_activation = False

        self.layout_estimation_gconvs = nn.ModuleList()
        hidden_dim = 288

        self.add_vert_feature = False

        if self.add_vert_feature:
            vert_feat_dim = 288
        else:
            vert_feat_dim = 0

        self.layout_estimation_vert_offset = nn.Linear(hidden_dim + 3, 3)

        for i in range(6):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3

            gconv = GraphConv(input_dim, hidden_dim, init="normal", directed=False)
            self.layout_estimation_gconvs.append(gconv)

        self.prediction_heads = nn.ModuleList()
        if self.size_cls_agnostic:
            self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=0))
        else:
            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                 mean_size_arr, num_proposal, 288,
                                                 emb_codes_dim=0))

        # Init
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.init_weights()
        self.init_bn_momentum()

    def forward(self, end_points):

        base_xyz = end_points['last_base_xyz']
        base_size = end_points['last_pred_size']
        obj_feature = end_points['last_object_featrue']

        # layout-vert
        refine_mesh = end_points['deep3d_meshes'][-1]
        vert_pos_padded = refine_mesh.verts_padded()
        vert_feature_padded = torch.transpose(end_points['refine_vert_feats'], 1, 2)
        vert_feature_proj = self.decoder_vert_featrue_proj(vert_feature_padded)

        # object
        object_pos = torch.cat([base_xyz, base_size], -1)
        query = self.decoder_obj_featrue_proj(obj_feature)

        # object cross attention
        object_feature_update, end_points['attention_weight_object'] = self.decoder[-1](query, vert_feature_proj, object_pos, vert_pos_padded)
        base_xyz, base_size = self.prediction_heads[-1](object_feature_update,
                                                  base_xyz=base_xyz,
                                                  end_points=end_points,
                                                  prefix='layout_refine_')

        # layout-vert cross attention
        vert_feature_update, end_points['attention_weight_layout'] = self.decoder[-2](vert_feature_proj, query, vert_pos_padded, object_pos)

        verts_padded_to_packed_idx = refine_mesh.verts_padded_to_packed_idx()
        vert_pos_packed = self._padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
        vert_feature_update = torch.transpose(vert_feature_update,1,2).contiguous()
        vert_feature_update_packed = self._padded_to_packed(vert_feature_update, verts_padded_to_packed_idx)

        # updatemesh
        first_layer_feats = [vert_feature_update_packed, vert_pos_packed]

        if(self.add_vert_feature):
            ori_vert_feature = self._padded_to_packed(end_points['refine_vert_feats'], verts_padded_to_packed_idx)
            first_layer_feats.append(ori_vert_feature)

        vert_feats = torch.cat(first_layer_feats, dim=1)

        ep = refine_mesh.edges_packed()

        # Run graph conv layers
        for gconv in self.layout_estimation_gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, ep))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

        # Predict a new mesh by offsetting verts
        if (self.use_activation):  ####if normalized mesh
            vert_offsets = torch.tanh(self.layout_estimation_vert_offset(vert_feats))
        else:
            vert_offsets = self.layout_estimation_vert_offset(vert_feats)

        refine_mesh = refine_mesh.offset_verts(vert_offsets)
        end_points['deep3d_meshes'].append(refine_mesh)

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _padded_to_packed(self, x, idx):
        D = x.shape[-1]
        idx = idx.view(-1, 1).expand(-1, D)

        x_packed = x.view(-1, D).gather(0, idx.to(x.device))

        return x_packed
