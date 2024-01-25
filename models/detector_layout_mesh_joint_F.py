import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import math
import numpy as np

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


class PositionEmbeddingRandomInit(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos.view(x.shape[0],pos.shape[1],-1)

def _padded_to_packed(x, idx):
    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
    x_packed = x.view(-1, D).gather(0, idx.to(x.device))

    return x_packed


def get_feature_map_pos():
    h = 16
    w = 32
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = - np.repeat(Phi, h, axis=0)
    X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
    Y = np.expand_dims(np.cos(Theta), 2)
    Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
    unit_map = np.concatenate([X, Z, Y], axis=2)
    return torch.from_numpy(np.array(unit_map,dtype=np.float32))


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
        self.num_decoder_layers = num_decoder_layers
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
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        self.decoder_layout_posembeds = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_layout_posembeds.append(PositionEmbeddingLearned(3, 288))

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
                                                                 full_size=True, hidden_dim=288, fuse_type='biproj')

        self.decoder_vert_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)

        from module.deep3dlayout.gcn_single_mesh import MeshRefinementStage
        self.layout_estimation_mesh_refinement = MeshRefinementStage(960, 288, 288, 6, gconv_init='normal',
                use_mhsa = True, use_rnn = False, mhsa_num_heads = 4, mhsa_hdim_factor = 2, use_pos_encoding = True, base_sphere_level = 3, level = 1, reduce_feats = True)

        self.featrue_token_flag = True
        self.random_masking = True
        if(self.featrue_token_flag):
            self.decoder_global_featrue_proj = nn.Conv1d(288, 288, kernel_size=1)
            self.decoder_img_feat_conv = nn.Conv2d(512, 288, 1)
            self.decoder_featrue_poseembs = PositionEmbeddingRandomInit(144)
            feat_pose_map = get_feature_map_pos()
            self.feat_pose_map = nn.Parameter(feat_pose_map.view(-1, 3).unsqueeze(0).repeat(4,1,1), requires_grad=False).to('cuda') # [4,512,3]
            self.decoder_feature_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_feature_posembeds.append(PositionEmbeddingLearned(3, 288))

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

        vert_pos_padded = meshes.verts_padded().detach().clone()

        vert_feats =  layout_ouput_dict['vert_feats']
        vert_feats = torch.transpose(vert_feats.view(bs, -1, 288), 1, 2)
        vert_feats_proj = self.decoder_vert_featrue_proj(vert_feats)

        if(self.featrue_token_flag):
            gloabl_featrue = layout_ouput_dict['gloabl_featrue']
            gloabl_featrue = self.decoder_img_feat_conv(gloabl_featrue)
            gloabl_featrue = self.decoder_global_featrue_proj(gloabl_featrue.flatten(2))
            query_joint = torch.cat([query, vert_feats_proj, key, gloabl_featrue], -1)
        else:
            query_joint = torch.cat([query, vert_feats_proj, key], -1)

        if (self.random_masking):
            meta_masks = inputs['meta_masks']
            meta_masks = torch.transpose(meta_masks,1,2).expand(-1,288,-1)
            constant_tensor = torch.ones_like(query_joint) * 0.01
            query_joint = query_joint * meta_masks[:,:,:query_joint.shape[2]] + constant_tensor * (1 - meta_masks[:,:,:query_joint.shape[2]])
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # position embs
            query_pos = torch.cat([base_xyz, base_size], -1)
            query_position_embs = self.decoder_self_posembeds[i](query_pos)
            key_position_embs = self.decoder_cross_posembeds[i](key_pos)
            layout_position_embs = self.decoder_layout_posembeds[i](vert_pos_padded)

            if (self.featrue_token_flag):
                feature_position_embs = self.decoder_feature_posembeds[i](self.feat_pose_map)
                position_embs_joint = torch.cat([query_position_embs, layout_position_embs, key_position_embs, feature_position_embs], -1)
            else:
                position_embs_joint = torch.cat([query_position_embs, layout_position_embs, key_position_embs], -1)
            position_embs_joint = position_embs_joint.permute(2, 0, 1)

            # transformer
            query_joint = query_joint.permute(2, 0, 1)
            query_joint = self.decoder[i](query_joint, pos = position_embs_joint)
            query_joint = query_joint.permute(1, 2, 0)

            # pred-object
            query = query_joint[:,:,0:self.num_proposal]
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        vert_feats_proj = torch.transpose(query_joint[:,:,self.num_proposal:(self.num_proposal+642)], 1, 2).contiguous()
        vert_feats_proj = vert_feats_proj.view(-1,288)

        subdivide = SubdivideMeshes()
        meshes, vert_feats_proj = subdivide(meshes, feats=vert_feats_proj)

        meshes, vert_feats = self.layout_estimation_mesh_refinement(layout_ouput_dict['img_feats'], meshes, vert_feats_proj,
                                                                    use_activation=False, reshaped_fh=False)

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
