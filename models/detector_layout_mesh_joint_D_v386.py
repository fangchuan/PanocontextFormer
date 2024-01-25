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
from .transformer_custorm3 import TransformerEncoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead, \
    ClsAgnosticPredictHead
from pytorch3d.ops import GraphConv, SubdivideMeshes
from pytorch3d.utils import ico_sphere
from module.deep3dlayout.gaf import gravity_projection

def _padded_to_packed(x, idx):
    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
    x_packed = x.view(-1, D).gather(0, idx.to(x.device))

    return x_packed

class GroupFreeDetectorDeep3D(nn.Module):
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
        self.num_decoder_layers = 6
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic
        self.emb_codes_dim = emb_codes_dim

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, width=self.width, image_feature_fusion=image_feature_fusion)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        if self.size_cls_agnostic:
            self.proposal_head = ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=self.emb_codes_dim)
        else:
            self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                             mean_size_arr, num_proposal, 288, emb_codes_dim=self.emb_codes_dim)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 285, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 285, kernel_size=1)

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerEncoderLayer(
                    288, nhead, dim_feedforward, dropout, activation
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            if self.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288, emb_codes_dim=self.emb_codes_dim))
            else:
                self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                         mean_size_arr, num_proposal, 288, emb_codes_dim=self.emb_codes_dim))


        # layout-estimation-model

        from module.deep3dlayout.deep3dlayout_model_p2e_single_mesh import Deep3DlayoutNetFuse
        print("using Deep3DlayoutNetFuse!")
        self.layout_estimation_single_mesh = Deep3DlayoutNetFuse(backbone='resnet18', decoder_type='rcnn_p2m_mhsa_pos_dual',
                                                                 full_size=True, hidden_dim=144, fuse_type='biproj')

        layout_estimation_reduce_feats_bottleneck1 = nn.Linear(960, 144)
        nn.init.normal_(layout_estimation_reduce_feats_bottleneck1.weight, mean=0.0, std=0.01)
        nn.init.constant_(layout_estimation_reduce_feats_bottleneck1.bias, 0)

        self.layout_estimation_reduce_feats_bottleneck = layout_estimation_reduce_feats_bottleneck1

        self.layout_estimation_projection = gravity_projection(use_mhsa=True, use_rnn=False, lfeats=960, num_heads=4,
                                                               hdim_factor=2, use_pos_encoding=True, verts_count=1538)
        # 2562

        self.decoder_vert_featrue_proj = nn.Conv1d(288, 285, kernel_size=1)

        hidden_dim = 144
        self.layout_estimation_vert_offset = nn.Linear(hidden_dim*2 + 3, 3)
        nn.init.zeros_(self.layout_estimation_vert_offset.weight)
        nn.init.constant_(self.layout_estimation_vert_offset.bias, 0)

        self.layout_estimation_gconvs = nn.ModuleList()

        vert_feat_dim = 144
        for i in range(6):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + vert_feat_dim + 3
            gconv = GraphConv(input_dim, hidden_dim + vert_feat_dim, init="normal", directed=False)
            self.layout_estimation_gconvs.append(gconv)

        # Init
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.init_weights()
        self.init_bn_momentum()

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points, image = inputs['image'], sample_pts = inputs['sample_pts'])

        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # Proposal
        # cluster_feature.shape [16, 288, 256]
        # cluster_xyz.shape [16, 256, 3]
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            # query.shape =  torch.Size([16, 288, 256])
            # key.shape =  torch.Size([16, 288, 1024])
            # points_features.shape =  torch.Size([16, 288, 1024])
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")


        bs = base_xyz.shape[0]
        layout_ouput_dict = self.layout_estimation_single_mesh(inputs)
        meshes = layout_ouput_dict['meshes']

        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()
        vert_pos_padded = meshes.verts_padded()
        vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
        vert_align_feats = self.layout_estimation_projection(layout_ouput_dict['img_feats'], vert_pos_padded, get_vertices=False, is_squeezed_h=False)
        vert_align_feats = _padded_to_packed(vert_align_feats, verts_padded_to_packed_idx)
        vert_align_feats = F.relu(self.layout_estimation_reduce_feats_bottleneck(vert_align_feats))
        vert_feats = torch.cat([vert_align_feats,layout_ouput_dict['vert_feats']], dim=1)
        vert_feats =  torch.transpose(vert_feats.view(bs, -1, 288), 1, 2)
        vert_feats_proj = self.decoder_vert_featrue_proj(vert_feats)
        ep = meshes.edges_packed()


        query = query.permute(0, 2, 1)
        query_with_pose = torch.cat([query, base_xyz],-1)

        vert_feats_proj = vert_feats_proj.permute(0, 2, 1)
        vert_feats_proj_with_pose = torch.cat([vert_feats_proj, vert_pos_padded],-1)

        key = key.permute(0, 2, 1)
        key_with_pose = torch.cat([key, key_pos],-1)

        query_joint = torch.cat([query_with_pose, vert_feats_proj_with_pose, key_with_pose], 1)
        query_joint = query_joint.permute(1, 0, 2)

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # transformer
            query_joint = self.decoder[i](query_joint)
            query_joint = query_joint.permute(1, 2, 0)

            # pred-object
            query = query_joint[:,:,0:self.num_proposal]
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            # base_xyz = base_xyz.detach().clone()
            # base_size = base_size.detach().clone()

            # Prediction layout
            vert_align_feats = query_joint[:,:,self.num_proposal:(self.num_proposal+1538)]
            vert_align_feats = torch.transpose(vert_align_feats, 1, 2).contiguous()
            vert_align_feats = vert_align_feats.view(-1,288)

            first_layer_feats = [vert_align_feats, vert_pos_packed]
            vert_feats = torch.cat(first_layer_feats, dim=1)

            # Run graph conv layers
            gconv = self.layout_estimation_gconvs[i]
            vert_feats_proj = F.relu(gconv(vert_feats, ep))
             
            query_joint = query_joint.permute(2, 0 ,1)

        vert_feats = torch.cat([vert_feats_proj, vert_pos_packed], dim=1)

        # Predict a new mesh by offsetting verts
        vert_offsets = self.layout_estimation_vert_offset(vert_feats)

        meshes = meshes.offset_verts(vert_offsets)

        end_points['deep3d_meshes'] = layout_ouput_dict['deep3d_meshes']
        end_points['deep3d_meshes'].append(meshes)

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
